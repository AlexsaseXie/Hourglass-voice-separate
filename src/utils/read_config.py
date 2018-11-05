from configobj import ConfigObj


class Config(object):
    "Read from a config file"

    def __init__(self, filename, if_gen=False):
        self.filename = filename
        config = ConfigObj(self.filename)
        self.config = config

        # meaningful comment about the experiment
        self.comment = config["comment"]

        # name of the model to be trained
        self.model_path = config["train"]["model_path"]

        # pretrained model flag
        self.preload_model = config["train"].as_bool("preload_model")

        # path of the pretrained model
        self.pretrain_modelpath = config["train"]["pretrain_model_path"]

        # number of epochs to train
        self.epochs = config["train"].as_int("num_epochs")

        # batch size for training
        self.batch_size = config["train"].as_int("batch_size")


        # Learning rate
        self.lr = config["train"].as_float("lr")

        # weight decay
        self.weight_decay = config["train"].as_float("weight_decay")

        # optimizer
        self.optim = config["train"]["optim"]

        # feature size
        self.feature_size = [ config["train"].as_int("feature_H"), config["train"].as_int("feature_W") ]

        # using gpu?
        self.use_gpu = config["train"].as_bool("use_gpu")

    def write_config(self, filename):
        """
        Write the details of the experiment in the form of a config file.
        This will be used to keep track of what experiments are running and 
        what parameters have been used.
        :return: 
        """
        # import json
        # with open(filename, 'w') as fp:
        #     json.dump(self.config.dict(), fp)
        self.config.filename = filename
        self.config.write()

    def get_all_attribute(self):
        """
        This function prints all the values of the attributes, just to cross
        check whether all the data types are correct.
        :return: Nothing, just printing
        """
        for attr, value in self.__dict__.items():
            print(attr, value)


if __name__ == "__main__":
    file = Config("config.yml")
    # file.write_config(sections, "config.yml")
    # print (file.config)
    print(file.write_config())