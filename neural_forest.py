"""
Build a neural forest.
Author:
    Sabareesh Mamidipaka
Date:
    12/25/2018
"""

import pandas as pd

class Neural_Forest(object):
    """
    Builds a neural forest taking in the data.
    Saves the different models built which can be used to predict later.
    """
    def __init__(self, name='my_model', n_repeat = 2):
        """
        params:
        name: name in which the models need to be saved.
        n_repeat: number of times each subset of features need to be used. default value is 2.
        """
        
        self.name = name
        self.n_repeat = n_repeat
        
    def fit(self, data:pd.DataFrame, layers: list, target_col:list, epochs=5, batch_size = 30):
        """
        params:
        data: Dataframe to be used for fitting the model.
        layers: the dimensions of the neural network. can be made dynamic to give greater variance.
        target_col: list of names of the target variables in the dataset.
        feature_subset: list of list of column numbers (each element being the subset of features for one neural network). 
                        Each element being a unique combination of features.
        epochs: number of iterations for optimization. 
        """

        self.layers = layers
        
        # moving the target labels to the end of the dataframe
        data = data[[c for c in data if c not in target_col] + target_col]
        
        # All unique combinations of features we will use to construct neural networks
        features_list = make_features_list(data.columns, layers[0], target_col)

        # n_repeat = number of neural networks we want with same set of features.
        # The neural networks will be different because we will be doing bagging each time.
        self.features_list = features_list*self.n_repeat

        # Looping through length of features list to build a neural network each time
        for index in range(len(self.features_list)):
            tf.reset_default_graph()
            with tf.name_scope('scope_%0i' %index):
                
                print(index)

                # Perform bagging and get the train and validation data
                train_feature, train_label, val_feature, val_label = bag_data(data, self.features_list[index], target_col)

                # Define a scaler to perform scaling for the data
                scaler = standardscaler(train_feature)
                
                # Initialize the weights and biases for the neural network
                nn_weights, nn_biases = initialize_NN(layers)

                # Define the necessary placeholders
                feature_pl = tf.placeholder('float',[None,train_feature.shape[1]], name='input_placeholder')
                feature_test_pl = tf.placeholder('float',[None,data.shape[1]-len(target_col)], name='test_placeholder')
                label_pl = tf.placeholder('float',[None,train_label.shape[1]])

                # Cost function is assumed to be mse
                cost = tf.losses.mean_squared_error(labels=label_pl, predictions=forward(feature_pl, nn_weights, nn_biases))
                optimizer = tf.train.AdamOptimizer().minimize(cost)

                with tf.Session() as sess:
                    
                    # slicing the data sent to the placeholder to keep only the required features 
                    feature_test = tf.stack([feature_test_pl[:,a] for a in self.features_list[index]], axis=1)

                    # prediction for the test data
                    prediction = tf.identity(forward(scaler.transform(feature_test), nn_weights, nn_biases), 'predict')
                    
                    # initialize all the variables
                    sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scope_%0i' %index)))
                    
                    # run the optimization step multiple times
                    for epoch in range(epochs):
                        epoch_loss = 0
                        
                        # Batch processing 
                        for i in range(int(train_feature.shape[0]/batch_size)):
                            epoch_x,epoch_y = train_feature[i*batch_size:(i+1)*batch_size,:], train_label[i*batch_size:(i+1)*batch_size,:]
                            sess.run(optimizer, feed_dict = {feature_pl:scaler.transform(epoch_x),label_pl:epoch_y})
                    
                    # evaluate the model by running it on the validation set
                    error = tf.identity(sess.run(cost, feed_dict = {feature_pl:scaler.transform(val_feature),label_pl:val_label}), 'val_error')
                    
                    # save the model
                    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scope_%0i' %index))
                    saver.save(sess, './'+self.name+'/'+self.name+'%i' %index)
    
        # run on entire training data to get fitted values
        self.fitted_values = self.predict(data.drop(target_col, axis=1))
    def predict(self, data):
        error = []
        result = []
        for index in range(len(self.features_list)):
            print(index)
            tf.reset_default_graph()
            with tf.Session() as sess:    
                saver = tf.train.import_meta_graph('./'+self.name+'/'+self.name+'%0i.meta' %index)
                saver.restore(sess, './'+ self.name+'/'+self.name+ '%i' %index)
                result.append(sess.run('scope_%i/predict:0' %index, feed_dict={'scope_%i/test_placeholder:0' %index: data}))
                error.append(sess.run('scope_%i/val_error:0'%index))
        accuracy = np.max(error)-error
        normalized = normalize(accuracy[:,np.newaxis], axis=0, norm='l1').ravel()
        final_prediction = np.sum(np.concatenate([result[i] for i in range(len(result))], axis=1)*normalized, axis=1)
        return final_prediction
