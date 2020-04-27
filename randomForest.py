import random
import csv
import numpy as np


# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent():

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString, isTraining):
        if isTraining:
            substract = 1
        else: 
            substract = 0
        numberArray = []
        for i in range(len(numberString) - substract):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray
                
    def registerInitialState(self):
        """
        Here we use it to load the training data.
        """

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i], True)
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        print "Creating random forest..."
        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of imtegers 0-3 indicating the action
        # taken in that state.

        # Desired columns list does not include the features where
        # all values are the same
        # Having a specific feature/column that values are the same
        # doesn't make any sense of including that feature in the tree.
        desired_columns = []
        for i in range(len(self.data[0])):
            s = set([d[i] for d in self.data])
            if len(s) == 2:
                desired_columns.append(i)
        
        self.best_classifier = None
        best_score = -1
        for i in range(10):  # Create 10 random forest and select the one with the highest score

            # Split the data into training and test set
            X_train, X_test, y_train, y_test = self.train_test_split(np.array(self.data), np.array(self.target), 0.20)

            # Create a random forest
            random_forest = self.random_forest(np.array(X_train), np.array(y_train), np.array(desired_columns))
            succs = 0
            for i in range(len(X_test)):
                m = self.get_move(X_test[i], random_forest)
                # print m[1], y_test[i]
                if m == y_test[i]:
                    succs += 1
            score = (float(succs)/len(y_test))*100

            # print len(y_test), ":", succs
            # print score

            if score > best_score:
                self.best_classifier = random_forest
                best_score = score

        print "Done, accuracy:", best_score

    # Here we just run the classifier to decide what to do
    def getAction(self, inpt):

        move = self.get_move(classifier.convertToArray(inpt, False), self.best_classifier)
        return move

    def train_test_split(self, data,targets, test_size):
        """
        Splits the data and targets into training and test data.
        test_size indicates the percentage of the test data.
        Return 4 elements: data train and test rows, targets train and test rows

        data: 2d array
        targets: array 
        test_size: float (between 0 and 1)
        """

        training_rows = random.sample(range(0,len(data)), int(len(data)*(1-test_size)))
        X_train = data[training_rows,]
        y_train = targets[training_rows,]

        test_rows = []
        for i in range(len(data)):
            if i not in training_rows:
                test_rows.append(i)
        
        X_test = data[test_rows,]
        y_test = targets[test_rows,]

        return X_train, X_test, y_train, y_test 

    def get_move(self,data, random_forest):
        """
        Find the predicted value of all decision trees in the 
        random forest and return the most seen value.

        random_forest: List of Decision trees
        data: Array
        """

        results = []
        for dtree in random_forest:
            results.append(self.classify_data(dtree, data))
        move = self.plurality_value(results)

        return move


    def classify_data(self, dtree, data):
        """
        Given a decision tree and the input data, predict and return
        the target value

        dtree: DecisionTree
        data: Array
        """

        # If current node is not a feature, then 
        # it is a prediction
        if dtree.feature_root == None:
            return dtree.prediction
        
        feature = dtree.feature_root
        column = data[feature]
        if column == 1: # 1 indicates the true branch
            return self.classify_data(dtree.true_branch, data)
        else:
            return self.classify_data(dtree.false_branch, data)


    def random_forest(self,data, target, desired_columns, number_of_trees=100):
        """
        Create and return a random forest of many decision trees.
        Desired columns array holds all of the features

        data: 2d array
        target: array
        desired_columns: array
        """

        # Create multiple decision trees
        # Each decision tree has different features and data
        random_forest = []
        for i in range(number_of_trees):

            # Select random features 
            features_column = random.sample(desired_columns, int(len(desired_columns)*70/100))            
            data_sample = data[:,features_column]

            # Select random data rows
            data_sample_rows = random.sample(range(0,len(data_sample)), int(len(data_sample)*70/100))
            data_sample = data_sample[data_sample_rows,]
            target_sample = target[data_sample_rows,]

            dtree = self.decision_tree(data_sample, features_column, target_sample, target_sample, 0)
            random_forest.append(dtree)

        return random_forest

    def decision_tree(self, data, features, target, parent_target, depth):
        """
        Create a decision tree.
        Returns a Node that is holding the decision tree in it.
        
        data: 2D array holding the data for each row
        features: Array of all features
        target: Array of all target 
        parent_target: Array of all targets of the parent node
        depth: Integer, holds the depth of the tree
        """
        
        # print len(data), len(features), len(target), len(parent_target)
        if depth > 3:
            if len(target) == 0:
                classification = self.plurality_value_prob(parent_target)
            else:
                classification = self.plurality_value_prob(target)
            tree = DecisionTree()
            tree.prediction = classification
            return tree

        # Data are can not be splitted more
        # so the decision tree picks the plurality value
        if len(target) == len(parent_target) and depth != 0:
            classification = self.plurality_value(target)
            tree = DecisionTree()
            tree.prediction = classification
            return tree

        # When no more rows exist, return the plurality value of the parent node
        if len(data) == 0:
            classification = self.plurality_value(parent_target)
            tree = DecisionTree()
            tree.prediction = classification
            return tree  
        elif len(set([d for d in target])) == 1:    # When all values of the remaining targets are the same return that value
            tree = DecisionTree()
            tree.prediction = target[0]
            return tree
        elif len(features) == 0:    # When no more features left to split the data, return the plurality value of the remaining targets
            classification = self.plurality_value_prob(target)
            tree = DecisionTree()
            tree.prediction = classification
            return tree

        # Get best feature, the one that splits the data the best possible way
        best_feature = self.most_important_feature(data,features,target)
        if best_feature == None:
            classification = self.plurality_value_prob(target)
            tree = DecisionTree()
            tree.prediction = classification
            return tree

        tree = DecisionTree()
        tree.feature_root = best_feature[1]
        value_of_best_feature = data[:,best_feature[0]]
        values_set = set([d for d in value_of_best_feature])

        # Split the data based on the best feature selected and create 2 branches
        # False branch is where the value of the feature is 0
        # True branch is where the value of the feature is 1
        false_branch = []
        false_branch_target = []
        true_branch = []
        true_branch_target = []
        for value in values_set:    
            for i in range(len(data)):
                if value == value_of_best_feature[i] and value == 0:
                    false_branch.append(np.array(data[i]))
                    false_branch_target.append(target[i])
                if value == value_of_best_feature[i] and value == 1:
                    true_branch.append(np.array(data[i]))
                    true_branch_target.append(target[i])

        # Remove best feature
        remain_features = np.delete(features,best_feature[0])

        # Create the 2 branches
        false_branch_tree = self.decision_tree(np.array(false_branch), remain_features, false_branch_target, np.array(target), depth+1)
        true_branch_tree = self.decision_tree(np.array(true_branch), remain_features, true_branch_target, np.array(target), depth+1)

        tree.false_branch = false_branch_tree
        tree.true_branch = true_branch_tree

        return tree
    
    def most_important_feature(self, data, features, target):
        """
        Calculate the information gain for the given features 
        and return the one with the highest information gain

        return Tuple of 2 integers.
        First integer indicates the position of the feature in the given list
        Second integer indicates the unique name of the feature from all of the features.
        """

        #Find entropy of parent, H(S)
        target_dict = dict()
        for t in target:
            if target_dict.get(t) == None:
                target_dict[t] = 0
            target_dict[t] += 1 
        
        p_targets = []
        for t in target_dict:
            p_targets.append(float(target_dict[t])/float(len(target)))

        log2 = np.log2(p_targets)
        parent_entropy = -1.0* np.sum(p_targets * log2) # = H(S)


        best_feature = None
        maximum_info_gain = -1000
        for i,feature in enumerate(features):   # Find information gain for each feature

            values_set = set([d[i] for d in data])
            data_per_features = [d[i] for d in data]

            feature_splitted = dict()
            for val in values_set:  # Calculate Si for each child
                split_feature = dict()
                split_feature["count"] = 0  
                for j,data_of_feature in enumerate(data_per_features):
                    if data_of_feature == val:
                        if split_feature.get(target[j]) == None:
                            split_feature[target[j]] = 0
                        split_feature[target[j]] += 1
                        split_feature["count"] += 1

                feature_splitted[val] = split_feature
                        
            entropies_of_childs = dict()
            for value in feature_splitted:  # Calculate entropy of all childs = H(Si)
                p_child = []
                size = feature_splitted.get(value).get("count")
                for val in feature_splitted.get(value):
                    if val != "count":
                        p_child.append(float(feature_splitted.get(value).get(val))/float(size))
                    
                p_child = np.array(p_child)
                log2_p_childs = np.log2(p_child)
                entropies_of_childs[value] = -1.0* np.sum(p_child * log2_p_childs)

            weighted_avg_of_entropied = 0
            for value in entropies_of_childs:   # Calculate the sigma of (Si/S)*H(Si)
                weighted_avg_of_entropied += float(feature_splitted.get(value).get("count"))/float(len(target))* entropies_of_childs.get(value)

            information_gain = parent_entropy - weighted_avg_of_entropied

            if information_gain > maximum_info_gain:
                maximum_info_gain = information_gain
                best_feature = i, feature
                
        return best_feature


    def plurality_value(self, targets):
        """
        Pick the most seen value from the targets

        targets: Array
        """
        unique = np.unique(targets)
        counts = []
        for u in unique:
            count = 0
            for target in targets:
                if u == target:
                    count += 1
            counts.append(count)

        populations = dict(zip(unique, counts))
        m = max(populations, key=populations.get)

        return m

    def plurality_value_prob(self, targets):
        """
        Creates a probability for each value in targets based on the number of times
        that value exists.
        Return a value from the targets based on their probabilities
        """
        unique = np.unique(targets)
        counts = []
        for u in unique:
            count = 0
            for target in targets:
                if u == target:
                    count += 1
            counts.append(count)

        populations = dict(zip(unique, counts))

        random_number = random.randint(1, len(targets))
        previous = 0
        for i in populations:
            if previous < random_number <= previous + populations[i]:
                return i
            else:
                previous += populations[i]

class DecisionTree:

    def __init__(self):
        self.feature_root = None
        self.true_branch = None
        self.false_branch = None
        self.prediction = None

# How to use
classifier = ClassifierAgent()
classifier.registerInitialState()
inpt = "1010010100000000000000000"
print classifier.getAction(inpt)