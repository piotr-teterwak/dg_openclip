import collections
import json
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from domainbed.datasets import get_dataset, split_dataset
from domainbed import algorithms
from domainbed.evaluator import Evaluator
from domainbed.lib import misc
from domainbed.lib import swa_utils
from domainbed.lib.query import Q
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import swad as swad_module


def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")


def train(test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None):
    logger.info("")

    #######################################################
    # setup dataset & loader
    #######################################################
    args.real_test_envs = test_envs  # for log
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams, algorithm_class)
    test_splits = []

    if target_env is not None:
        testenv_name = f"te_{dataset.environments[target_env]}"
        logger.info(f"Target env = {target_env}")
    elif args.in_domain:
        testenv_name = f"te_id"
    else:
        testenv_properties = [str(dataset.environments[i]) for i in test_envs]
        testenv_name = "te_" + "_".join(testenv_properties)

    logger.info(
        "Testenv name escaping {} -> {}".format(testenv_name, testenv_name.replace(".", ""))
    )
    testenv_name = testenv_name.replace(".", "")
    logger.info(f"Test envs = {test_envs}, name = {testenv_name}")

    n_envs = len(dataset)
    train_envs = sorted(set(range(n_envs)) - set(test_envs))
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=np.int)
    if not args.in_domain:
        batch_sizes[test_envs] = 0
    batch_sizes = batch_sizes.tolist()

    logger.info(f"Batch sizes for each domain: {batch_sizes} (total={sum(batch_sizes)})")

    # calculate steps per epoch
    steps_per_epochs = [
        len(env) / batch_size
        for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    steps_per_epoch = min(steps_per_epochs)
    # epoch is computed by steps_per_epoch
    prt_steps = ", ".join([f"{step:.2f}" for step in steps_per_epochs])
    logger.info(f"steps-per-epoch for each domain: {prt_steps} -> min = {steps_per_epoch:.2f}")

    # setup loaders
    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=batch_size,
            num_workers=dataset.N_WORKERS,
        )
        for (env, env_weights), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    # setup eval loaders
    eval_loaders_kwargs = []
    for i, (env, _) in enumerate(in_splits + out_splits + test_splits):
        batchsize = hparams["test_batchsize"]
        loader_kwargs = {"dataset": env, "batch_size": batchsize, "num_workers": dataset.N_WORKERS}
        if args.prebuild_loader:
            loader_kwargs = FastDataLoader(**loader_kwargs)
        eval_loaders_kwargs.append(loader_kwargs)

    eval_weights = [None for _, weights in (in_splits + out_splits + test_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
    eval_loader_names += ["env{}_inTE".format(i) for i in range(len(test_splits))]
    eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))
    #######################################################
    # setup algorithm (model)
    #######################################################
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    )

    algorithm.cuda()

    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    #######################################################
    # start training loop
    #######################################################
    evaluator = Evaluator(
        test_envs,
        eval_meta,
        n_envs,
        logger,
        evalmode=args.evalmode,
        debug=args.debug,
        target_env=target_env,
        dump_scores = args.dump_scores,
        dump_similarities = args.dump_similarities,
        out_dir = args.out_dir
    )

    swad = None
    if hparams["swad"]:
        swad_algorithm = swa_utils.AveragedModel(algorithm)
        if args.mpa:
            swad_cls = getattr(swad_module, "MPA")
        else:
            swad_cls = getattr(swad_module, "LossValley")
        swad = swad_cls(evaluator, **hparams.swad_kwargs)

    last_results_keys = None
    records = []
    epochs_path = args.out_dir / "results.jsonl"

    if args.evaluate:

        #checkpoint = torch.load(args.resume_path)
        #if swad:
        #    algorithm = swa_utils.AveragedModel(algorithm)
        #algorithm.load_state_dict(checkpoint['model_dict'])
        #in_key = "train_out"
        #tr_val_best_indomain = records.argmax("train_out")[in_key]

        #ret = {

        #  "test-domain validation": te_val_best,
        #"training-domain validation": tr_val_best,
        #  "last": last,
        #  "last (inD)": last_indomain,
        #  "training-domain validation (inD)": tr_val_best_indomain,
        #}



        # swad_algorithm = swad.get_final_model()
        # if hparams["freeze_bn"] is False:
        #     n_steps = 500 if not args.debug else 10
        #     logger.warning(f"Update SWAD BN statistics for {n_steps} steps ...")
        #     swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, n_steps)

        # logger.warning("Evaluate SWAD ...")
        accuracies, summaries = evaluator.evaluate(algorithm)
        results = {**summaries, **accuracies}
        results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
        row = misc.to_row([results[key] for key in results_keys if key in results])
        logger.info(row)

        #ret["SWAD"] = results["test_in"]
        #ret["SWAD (inD)"] = results[in_key]
        ret = {}
        ret['OOD'] =  0.8 * results["test_in"] + 0.2 * results["test_out"]
        ret['ID'] = results["train_out"]

        for k, acc in ret.items():
            logger.info(f"{k} = {acc:.3%}")


        return ret, records

    for step in range(n_steps):
        step_start_time = time.time()
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]
        batches_dictlist = next(train_minibatches_iterator)
        # batches: {data_key: [env0_tensor, ...], ...}
        batches = misc.merge_dictlist(batches_dictlist)
        # to device
        batches = {
            key: [tensor.cuda() for tensor in tensorlist] for key, tensorlist in batches.items()
        }

        inputs = {**batches, "step": step}
        step_vals = algorithm.update(**inputs)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        if swad:
            # swad_algorithm is segment_swa for swad
            swad_algorithm.update_parameters(algorithm, step=step)

        if step % checkpoint_freq == 0:
            results = {
                "step": step,
                "epoch": step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            eval_start_time = time.time()
            accuracies, summaries = evaluator.evaluate(algorithm)
            results["eval_time"] = time.time() - eval_start_time

            # results = (epochs, loss, step, step_time)
            results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
            # merge results
            results.update(summaries)
            results.update(accuracies)
            val_accurracy_keys = [k.split('_')[0] for k in accuracies.keys()]
            if len(test_envs) > 0:
                val_accurracy_keys = ['{}_out'.format(k) for k in val_accurracy_keys if str(test_envs[0]) not in k]
            else:
                val_accurracy_keys = ['{}_out'.format(k) for k in val_accurracy_keys]
            val_accurracy = np.mean([accuracies[k] for k in val_accurracy_keys])
            is_best = False
            if val_accurracy > algorithm.best_val_acc:
                algorithm.best_val_acc = val_accurracy
                is_best = True

            # print
            if results_keys != last_results_keys:
                logger.info(misc.to_row(results_keys))
                last_results_keys = results_keys
            logger.info(misc.to_row([results[key] for key in results_keys]))
            records.append(copy.deepcopy(results))

            # update results to record
            results.update({"hparams": dict(hparams), "args": vars(args)})

            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True, default=json_handler) + "\n")

            checkpoint_vals = collections.defaultdict(lambda: [])

            writer.add_scalars_with_prefix(summaries, step, f"{testenv_name}/summary/")
            writer.add_scalars_with_prefix(accuracies, step, f"{testenv_name}/all/")

            if is_best:
                ckpt_dir = args.out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)

                if args.in_domain:
                    test_env_str = test_envs
                else:
                    test_env_str = ",".join(map(str, test_envs))
                filename = "TE{}_best.pth".format(test_env_str)
                if not args.in_domain and len(test_envs) > 1 and target_env is not None:
                    train_env_str = ",".join(map(str, train_envs))
                    filename = f"TE{target_env}_TR{train_env_str}_best.pth"
                path = ckpt_dir / filename


                save_dict = {
                    "args": vars(args),
                    "model_hparams": dict(hparams),
                    "test_envs": test_envs,
                    "model_dict": algorithm.cpu().state_dict(),
                }
                algorithm.cuda()
                if not args.debug:
                    torch.save(save_dict, path)
                else:
                    logger.debug("DEBUG Mode -> no save (org path: %s)" % path)

            # swad
            if swad and step > hparams["linear_steps"]:
                def prt_results_fn(results, avgmodel):
                    step_str = f" [{avgmodel.start_step}-{avgmodel.end_step}]"
                    row = misc.to_row([results[key] for key in results_keys if key in results])
                    logger.info(row + step_str)

                swad.update_and_evaluate(
                    swad_algorithm, results["train_out"], results["tr_outloss"], prt_results_fn
                )

                if hasattr(swad, "dead_valley") and swad.dead_valley:
                    logger.info("SWAD valley is dead -> early stop !")
                    break

                swad_algorithm = swa_utils.AveragedModel(algorithm)  # reset
            if args.auto_lr and step > hparams["linear_steps"]:
                algorithm.scheduler.step(results["tr_outloss"])
                algorithm.lr_schedule.append(algorithm.scheduler._last_lr)
                if len(algorithm.lr_schedule) > 1:
                    if algorithm.lr_schedule[-1] != algorithm.lr_schedule[-2]:
                    #if True:
                        algorithm.lr_schedule_changes += 1
                    if algorithm.lr_schedule_changes == 3:
                            break


        if step % args.tb_freq == 0:
            # add step values only for tb log
            writer.add_scalars_with_prefix(step_vals, step, f"{testenv_name}/summary/")

    # find best
    logger.info("---")
    records = Q(records)
    te_val_best = records.argmax("test_out")["test_in"]
    tr_val_best = records.argmax("train_out")["test_in"]
    last = records[-1]["test_in"]

    in_key = "train_out"
    tr_val_best_indomain = records.argmax("train_out")[in_key]
    last_indomain = records[-1][in_key]

    # NOTE for clearity, report only training-domain validation results.
    ret = {
        #  "test-domain validation": te_val_best,
        "training-domain validation": tr_val_best,
        #  "last": last,
        #  "last (inD)": last_indomain,
        #  "training-domain validation (inD)": tr_val_best_indomain,
    }

    # Evaluate SWAD
    if swad:
        swad_algorithm = swad.get_final_model()
        if hparams["freeze_bn"] is False:
            n_steps = 500 if not args.debug else 10
            logger.warning(f"Update SWAD BN statistics for {n_steps} steps ...")
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, n_steps)

        logger.warning("Evaluate SWAD ...")
        accuracies, summaries = evaluator.evaluate(swad_algorithm)
        results = {**summaries, **accuracies}
        start = swad_algorithm.start_step
        end = swad_algorithm.end_step
        step_str = f" [{start}-{end}]  (N={swad_algorithm.n_averaged})"
        row = misc.to_row([results[key] for key in results_keys if key in results]) + step_str
        logger.info(row)

        ret["SWAD"] = results["test_in"]
        ret["SWAD (inD)"] = results[in_key]

        ckpt_dir = args.out_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        test_env_str = ",".join(map(str, test_envs))
        filename = "TE{}_best.pth".format(test_env_str)
        if len(test_envs) > 1 and target_env is not None:
            train_env_str = ",".join(map(str, train_envs))
            filename = f"TE{target_env}_TR{train_env_str}_best.pth"
        path = ckpt_dir / filename


        save_dict = {
            "args": vars(args),
            "model_hparams": dict(hparams),
            "test_envs": test_envs,
            "model_dict": swad_algorithm.cpu().state_dict(),
        }
        torch.save(save_dict, path)

    for k, acc in ret.items():
        logger.info(f"{k} = {acc:.3%}")






    ##### SECOND STAGE####################
    if args.full_data:
        args.holdout_fraction = 0.0
        dataset, in_splits, _ = get_dataset(test_envs, args, hparams, algorithm_class)
        test_splits = []
        out_splits = []

        if target_env is not None:
            testenv_name = f"te_{dataset.environments[target_env]}"
            logger.info(f"Target env = {target_env}")
        else:
            testenv_properties = [str(dataset.environments[i]) for i in test_envs]
            testenv_name = "te_" + "_".join(testenv_properties)

        logger.info(
            "Testenv name escaping {} -> {}".format(testenv_name, testenv_name.replace(".", ""))
        )
        testenv_name = testenv_name.replace(".", "")
        logger.info(f"Test envs = {test_envs}, name = {testenv_name}")

        n_envs = len(dataset)
        train_envs = sorted(set(range(n_envs)) - set(test_envs))
        iterator = misc.SplitIterator(test_envs)
        batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=np.int)

        batch_sizes[test_envs] = 0
        batch_sizes = batch_sizes.tolist()

        logger.info(f"Batch sizes for each domain: {batch_sizes} (total={sum(batch_sizes)})")

        # calculate steps per epoch
        steps_per_epochs = [
            len(env) / batch_size
            for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
        ]
        steps_per_epoch = min(steps_per_epochs)
        # epoch is computed by steps_per_epoch
        prt_steps = ", ".join([f"{step:.2f}" for step in steps_per_epochs])
        logger.info(f"steps-per-epoch for each domain: {prt_steps} -> min = {steps_per_epoch:.2f}")

        # setup loaders
        train_loaders = [
            InfiniteDataLoader(
                dataset=env,
                weights=env_weights,
                batch_size=batch_size,
                num_workers=dataset.N_WORKERS,
            )
            for (env, env_weights), batch_size in iterator.train(zip(in_splits, batch_sizes))
        ]

        # setup eval loaders
        eval_loaders_kwargs = []
        for i, (env, _) in enumerate(in_splits + out_splits + test_splits):
            batchsize = hparams["test_batchsize"]
            loader_kwargs = {"dataset": env, "batch_size": batchsize, "num_workers": dataset.N_WORKERS}
            if args.prebuild_loader:
                loader_kwargs = FastDataLoader(**loader_kwargs)
            eval_loaders_kwargs.append(loader_kwargs)

        eval_weights = [None for _, weights in (in_splits + out_splits + test_splits)]
        eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
        eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
        eval_loader_names += ["env{}_inTE".format(i) for i in range(len(test_splits))]
        eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))

        #######################################################
        # setup algorithm (model)
        #######################################################
        algorithm = algorithm_class(
            dataset.input_shape,
            dataset.num_classes,
            len(dataset) - len(test_envs),
            hparams,
        )

        algorithm.cuda()

        n_params = sum([p.numel() for p in algorithm.parameters()])
        logger.info("# of params = %d" % n_params)

        train_minibatches_iterator = zip(*train_loaders)
        checkpoint_vals = collections.defaultdict(lambda: [])

        #######################################################
        # start training loop
        #######################################################
        evaluator = Evaluator(
            test_envs,
            eval_meta,
            n_envs,
            logger,
            evalmode=args.evalmode,
            debug=args.debug,
            target_env=target_env,
            dump_scores = args.dump_scores,
            dump_similarities = args.dump_similarities,
            out_dir = args.out_dir
        )
        if hparams["swad"]:
            first_stage_converged = swad.is_converged
            swad_algorithm = swa_utils.AveragedModel(algorithm)
            swad_cls = getattr(swad_module, "SecondStage")
            swad = swad_cls(evaluator, start, end, first_stage_converged,  **hparams.swad_kwargs)
            second_stage_steps = n_steps
        else:
            second_stage_steps = records.argmax("train_out")["step"] + 1

        last_results_keys = None
        records = []
        epochs_path = args.out_dir / "results.jsonl"

        for step in range(second_stage_steps):
            step_start_time = time.time()
            # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]
            batches_dictlist = next(train_minibatches_iterator)
            # batches: {data_key: [env0_tensor, ...], ...}
            batches = misc.merge_dictlist(batches_dictlist)
            # to device
            batches = {
                key: [tensor.cuda() for tensor in tensorlist] for key, tensorlist in batches.items()
            }

            inputs = {**batches, "step": step}
            step_vals = algorithm.update(**inputs)
            for key, val in step_vals.items():
                checkpoint_vals[key].append(val)
            checkpoint_vals["step_time"].append(time.time() - step_start_time)

            if swad:
                # swad_algorithm is segment_swa for swad
                swad_algorithm.update_parameters(algorithm, step=step)

            if step % checkpoint_freq == 0:
                results = {
                    "step": step,
                    "epoch": step / steps_per_epoch,
                }

                for key, val in checkpoint_vals.items():
                    results[key] = np.mean(val)

                eval_start_time = time.time()
                accuracies, summaries = evaluator.evaluate(algorithm)
                results["eval_time"] = time.time() - eval_start_time

                # results = (epochs, loss, step, step_time)
                results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
                # merge results
                results.update(summaries)
                results.update(accuracies)
                val_accurracy_keys = [k.split('_')[0] for k in accuracies.keys()]
                val_accuracy_keys = ['{}_out'.format(k) for k in val_accurracy_keys if str(test_envs[0]) not in k]
                val_accuracy = np.mean([accuracies[k] for k in val_accuracy_keys])
                is_best = False
                if val_accuracy > algorithm.best_val_acc:
                    algorithm.best_val_acc = val_accuracy
                    is_best = True

                # print
                if results_keys != last_results_keys:
                    logger.info(misc.to_row(results_keys))
                    last_results_keys = results_keys
                logger.info(misc.to_row([results[key] for key in results_keys]))
                records.append(copy.deepcopy(results))

                # update results to record
                results.update({"hparams": dict(hparams), "args": vars(args)})

                with open(epochs_path, "a") as f:
                    f.write(json.dumps(results, sort_keys=True, default=json_handler) + "\n")

                checkpoint_vals = collections.defaultdict(lambda: [])

                writer.add_scalars_with_prefix(summaries, step, f"{testenv_name}/summary/")
                writer.add_scalars_with_prefix(accuracies, step, f"{testenv_name}/all/")

                if is_best:
                    ckpt_dir = args.out_dir / "checkpoints"
                    ckpt_dir.mkdir(exist_ok=True)

                    test_env_str = ",".join(map(str, test_envs))
                    filename = "TE{}_best.pth".format(test_env_str)
                    if len(test_envs) > 1 and target_env is not None:
                        train_env_str = ",".join(map(str, train_envs))
                        filename = f"TE{target_env}_TR{train_env_str}_best.pth"
                    path = ckpt_dir / filename


                    save_dict = {
                        "args": vars(args),
                        "model_hparams": dict(hparams),
                        "test_envs": test_envs,
                        "model_dict": algorithm.cpu().state_dict(),
                    }
                    algorithm.cuda()
                    if not args.debug:
                        torch.save(save_dict, path)
                    else:
                        logger.debug("DEBUG Mode -> no save (org path: %s)" % path)

                # swad
                if swad and step > hparams["linear_steps"]:
                    def prt_results_fn(results, avgmodel):
                        step_str = f" [{avgmodel.start_step}-{avgmodel.end_step}]"
                        row = misc.to_row([results[key] for key in results_keys if key in results])
                        logger.info(row + step_str)

                    swad.update_and_evaluate(
                        swad_algorithm, prt_results_fn
                    )

                    if hasattr(swad, "dead_valley") and swad.dead_valley:
                        logger.info("SWAD valley is dead -> early stop !")
                        break

                    swad_algorithm = swa_utils.AveragedModel(algorithm)  # reset

            if step % args.tb_freq == 0:
                # add step values only for tb log
                writer.add_scalars_with_prefix(step_vals, step, f"{testenv_name}/summary/")

        # find best
        logger.info("---")
        records = Q(records)
        last = records[-1]["test_in"]

        in_key = "train_out"
        last_indomain = records[-1][in_key]

        # NOTE for clearity, report only training-domain validation results.
        ret = {
            #  "test-domain validation": te_val_best,
            #"training-domain validation": tr_val_best,
              "last": last,
            #  "last (inD)": last_indomain,
            #  "training-domain validation (inD)": tr_val_best_indomain,
        }

        # Evaluate SWAD
        if swad:
            swad_algorithm = swad.get_final_model()
            if hparams["freeze_bn"] is False:
                n_steps = 500 if not args.debug else 10
                logger.warning(f"Update SWAD BN statistics for {n_steps} steps ...")
                swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, n_steps)

            logger.warning("Evaluate SWAD ...")
            accuracies, summaries = evaluator.evaluate(swad_algorithm)
            results = {**summaries, **accuracies}
            start = swad_algorithm.start_step
            end = swad_algorithm.end_step
            step_str = f" [{start}-{end}]  (N={swad_algorithm.n_averaged})"
            row = misc.to_row([results[key] for key in results_keys if key in results]) + step_str
            logger.info(row)

            ret["SWAD"] = results["test_in"]
            ret["SWAD (inD)"] = results[in_key]
            state_dict = swad_algorithm.cpu().state_dict()
        else:
            state_dict = algorithm.cpu().state_dict()

        ckpt_dir = args.out_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        test_env_str = ",".join(map(str, test_envs))
        filename = "TE{}_best.pth".format(test_env_str)
        if len(test_envs) > 1 and target_env is not None:
            train_env_str = ",".join(map(str, train_envs))
            filename = f"TE{target_env}_TR{train_env_str}_best_scnd_stage.pth"
        path = ckpt_dir / filename


        save_dict = {
            "args": vars(args),
            "model_hparams": dict(hparams),
            "test_envs": test_envs,
            "model_dict": state_dict
        }
        torch.save(save_dict, path)

        for k, acc in ret.items():
            logger.info(f"{k} = {acc:.3%}")


    return ret, records
