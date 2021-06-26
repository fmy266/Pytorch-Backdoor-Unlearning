if __name__ == "__main__":
    import torch
    import trojanvision
    from trojanvision.utils import summary
    import unlearning_backdoor
    import knowledge_distillation
    import forget
    import variance
    epoch = 50
    lr = 0.01  # learning rate
    poison_percent = 0.05
    # cifar10, cifar100, mnist, fashionmnist
    # badnet trojannn imc
    # 3,3      5,5      7,7
    for dataset_name in ["cifar10","cifar100","mnist","fashionmnist"]:
        for trigger_size in [5, 7]:
            for attack_method in ["badnet", "trojannn", "imc"]:
                env = trojanvision.environ.create(num_gpus = 2, verbose = True)
                dataset = trojanvision.datasets.create(dataset_name, folder_path="..//data")
                model = trojanvision.models.create(dataset=dataset, model="resnet18")
                # 测试不同trigger的大小影响
                flag = True if attack_method=="badnet" else False
                mark = trojanvision.marks.create(dataset=dataset, random_init=True, random_pos = flag,
                                                 mark_height = trigger_size, mark_width = trigger_size)
                attack = trojanvision.attacks.create(save = True, poison_percent = poison_percent,
                                                     dataset=dataset, model=model, mark=mark,
                                                     attack_name=attack_method) # badnet, trojannn, imc
                # trainer = trojanvision.trainer.create(dataset=dataset, model=model, epoch = epoch, lr = lr,
                #                                       validate_interval = 5, save = True)
                attack.load()
                knowledge_distillation_method = knowledge_distillation.KnowledgeDistillation(model, dataset, mark, attack)
                knowledge_distillation_method.run()

    # if env['verbose']: # report detailed information about training
    #     summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    #
    # attack.attack(**trainer)

#######################################################################################

    # attack.load()
    # defense_name = "neural_cleanse"
    # defense = trojanvision.defenses.create(dataset=dataset, model=model, attack=attack, defense_name=defense_name)
    # trainer = trojanvision.trainer.create(dataset=dataset, model=model, epoch = 1, lr = lr,
    #                                       validate_interval = 1, save = False, clean_image_num = 2500)
    # if env['verbose']:
    #     summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack, defense=defense)
    # defense.detect(**trainer)

#######################################################################################

    # gan_based method for plot
    # attack.load()
    # for name, module in model.named_modules():
    #     if name == "features.layer1.1.conv1":
    #         module.register_forward_hook(visual.score_cam_hook_func)
    # unlearning_method = unlearning_backdoor.UnlearningBackdoor(model, dataset, mark, attack, 0.25, 1)
    # unlearning_method.run()

#######################################################################################

    # attack.load()
    # defense_name = "fine_pruning"
    # defense = trojanvision.defenses.create(dataset=dataset, model=model, attack=attack, defense_name=defense_name, prune_ratio = 0.99)
    # trainer = trojanvision.trainer.create(dataset=dataset, model=model, epoch = 10, lr = lr,
    #                                       validate_interval = 1, save = False, clean_image_num = 2500)
    # defense.detect(**trainer)

 #######################################################################################

    # unlearning_based
    # for ratio in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    #     for alpha in [100]:
    #         attack.load()
    #         unlearning_method = unlearning_backdoor.UnlearningBackdoor(model, dataset, mark, attack, ratio, alpha)
    #         unlearning_method.run()
    #
    # for ratio in [0.25]:
    #     for alpha in [1e-4,1e-3,1e-2,1e-1,0,1,10,100,1000,1e4,1e5]:
    #         attack.load()
    #         unlearning_method = unlearning_backdoor.UnlearningBackdoor(model, dataset, mark, attack, ratio, alpha)
    #         unlearning_method.run()

 #######################################################################################

    # knowledge distillation
    # attack.load()
    # knowledge_distillation_method = knowledge_distillation.KnowledgeDistillation(model, dataset, mark, attack)
    # knowledge_distillation_method.run()

#######################################################################################
    # importance-based
    # for training_epoch in [100]:
    #     for start_reserved_ratio in [0.20]: # 0.20
    #         for end_reserved_ratio in [0.3, 0.4, 0.5, 0.6, 0.7]: # 0.5 0.75 1.0
    #             for sample_prob in [0.1]: # 0.5
    #                 for alpha in [3e-3]: # 1e-1,1e-2,1e-3
    #                     attack.load()
    #                     forget_method = forget.Forget(model, dataset, mark, attack)
    #                     forget_method.training_epoch = training_epoch
    #                     forget_method.start_reserved_ratio = start_reserved_ratio
    #                     forget_method.end_reserved_ratio = end_reserved_ratio
    #                     forget_method.sample_prob = sample_prob
    #                     forget_method.alpha = alpha
    #                     forget_method.run()
    #
    # for training_epoch in [100]:
    #     for start_reserved_ratio in [0.20]: # 0.20
    #         for end_reserved_ratio in [0.50]: # 0.5 0.75 1.0
    #             for sample_prob in [0.1, 0.2, 0.3, 0.4, 0.5]: # 0.5
    #                 for alpha in [3e-3]: # 1e-1,1e-2,1e-3
    #                     attack.load()
    #                     forget_method = forget.Forget(model, dataset, mark, attack)
    #                     forget_method.training_epoch = training_epoch
    #                     forget_method.start_reserved_ratio = start_reserved_ratio
    #                     forget_method.end_reserved_ratio = end_reserved_ratio
    #                     forget_method.sample_prob = sample_prob
    #                     forget_method.alpha = alpha
    #                     forget_method.run()
    #
    #
    # for training_epoch in [100]:
    #     for start_reserved_ratio in [0.20]: # 0.20
    #         for end_reserved_ratio in [0.50]: # 0.5 0.75 1.0
    #             for sample_prob in [0.1]: # 0.5
    #                 for alpha in [1e-2, 7e-3, 5e-3, 3e-3, 1e-3]: # 1e-1,1e-2,1e-3
    #                     attack.load()
    #                     forget_method = forget.Forget(model, dataset, mark, attack)
    #                     forget_method.training_epoch = training_epoch
    #                     forget_method.start_reserved_ratio = start_reserved_ratio
    #                     forget_method.end_reserved_ratio = end_reserved_ratio
    #                     forget_method.sample_prob = sample_prob
    #                     forget_method.alpha = alpha
    #                     forget_method.run()

#######################################################################################
    # variance_baed
    # for start_reserved_ratio in [0.05]: # 0.20
    #     for end_reserved_ratio in [0.10, 0.20, 0.50]: # 0.5 0.75 1.0
    #         for sample_prob in [0.01,0.02,0.03,0.04,0.05]: # 0.5
    #             for alpha in [1e5,1e4,1e3,1e2,1e1,1,1e-1,1e-2,1e-3,1e-4]: # 1e-1,1e-2,1e-3
    #                 attack.load()
    #                 variance_method = variance.VarPruning(model, dataset, mark, attack)
    #                 variance_method.start_reserved_ratio = start_reserved_ratio
    #                 variance_method.end_reserved_ratio = end_reserved_ratio
    #                 variance_method.sample_prob = sample_prob
    #                 variance_method.alpha = alpha
    #                 variance_method.run()

