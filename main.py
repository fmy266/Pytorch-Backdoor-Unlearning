if __name__ == "__main__":
    import torch
    import trojanvision
    from trojanvision.utils import summary
    import unlearning_backdoor

    # attacking ....
    epoch = 50 # the accuracy of the model trained 50 epoch probably higher than 80%
    lr = 0.01 # learning rate
    poison_percent = 0.05 # This is widely-used in most papers (typical poison percert)

    env = trojanvision.environ.create(num_gpus = 2, verbose = True)
    dataset = trojanvision.datasets.create("cifar10", folder_path="..//data")
    model = trojanvision.models.create(dataset=dataset, model="resnet18")
    # the trigger with a random position is more threat than the trigger with a fixed position and therefore we adopt the setting.
    mark = trojanvision.marks.create(dataset=dataset, random_init=True, random_pos = True, 
                                     mark_height = 3, mark_width = 3)
    attack = trojanvision.attacks.create(save = True, poison_percent = poison_percent,
                                         dataset=dataset, model=model, mark=mark,
                                         attack_name="badnet")
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, epoch = epoch, lr = lr,
                                          validate_interval = 5, save = True)

    if env['verbose']: # report detailed information about training
        summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    
    attack.attack(**trainer)

    # defense using our method
    # 0.25 means holding 0.25% test set, which equals to 0.05% train dataset
    defense = unlearning_backdoor.UnlearningBackdoor(model, dataset, mark, attack, 0.25, 100) 
    defense.run()