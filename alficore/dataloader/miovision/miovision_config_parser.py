import os, yaml, sys


class ConfigParser:
    def __init__(self, **entries):
        self.train_ground_truth_file = None
        self.root_directory = None
        self.test_image_names = None
        self.correct_val_ground_truth_file = None
        self.rescale_dimension = 100
        self.random_seed = 42
        self.train_split = 0.8
        self.validation_split = 0.2
        self.generate_ranger_bounds = False
        self.inject_fault = False
        self.shuffle_dataset = True
        self.batch_size = 1
        self.train_class_indices_checkpoint = None
        self.val_indices_fault_inj_weights_checkpoint = None
        self.learning_rate = 0.0065
        self.momentum = 0.9
        self.weight_decay = 0.0004
        self.patience = 5
        self.num_epochs = 1
        self.train = False
        self.evaluate = False
        self.val_correct = False
        self.separate_correct_val = False
        self.train_uniform_class_dist = False
        self.num_classes = 11
        self.num_test_images = 1
        self.dataset_name = None
        self.model_name = None
        self.ranger_file_name = None
        self.generate_ranger_bounds_classes = False
        self.faults_per_inference = None
        self.fault_epochs = None
        self.fault_type = 'weights_injs'
        self.number_of_images_each_class = 0
        self.bit_range = None
        self.neuron_inj_class_id = 0
        self.__dict__.update(entries)

def load_scenario(conf_location):
        """
        Load content of scenario file (yaml)
        :param conf_location: relative path to miovision configuration,
        default is 'dataset_configs/miovision_config.yml' 
        :return: dict from yaml file
        """
        fileDir = os.path.dirname(sys.argv[0])
        # fileDir = os.path.dirname(os.path.realpath(__file__))
        # Obtain the current working directory
        config_file = os.path.join(fileDir, conf_location)
        document = open(config_file)
        scenario = yaml.safe_load(document)
        return scenario  
        