import numpy as np
import pandas as pd
import warnings
import math
import os
from sklearn.model_selection import train_test_split


class MaxSizeModel:

    def __init__(self, num_partition=-1, num_rules=2, max_rule_size=2, data_fidelity=10, weight_feature=1,
                 solver="open-wbo", rule_type="DNF", work_dir=".", time_out=1024):
        self.num_partition = num_partition
        self.num_rules = num_rules
        self.dataFidelity = data_fidelity
        self.weightFeature = weight_feature
        self.solver = solver
        self.ruleType = rule_type
        self.workDir = work_dir
        self.verbose = False  # not necessary
        self.trainingError = 0
        self.selectedFeatureIndex = []
        self.columns = []
        self.timeOut = time_out
        self.max_rule_size = max_rule_size

    def discretize(self, file, categorical_column_index=None, column_separator=",", frac_present=0.9, num_threshold=4):

        if categorical_column_index is None:
            categorical_column_index = []
        # Quantile probabilities
        quant_prob = np.linspace(1. / (num_threshold + 1.), num_threshold / (num_threshold + 1.), num_threshold)
        # List of categorical columns
        if type(categorical_column_index) is pd.Series:
            categorical_column_index = categorical_column_index.tolist()
        elif type(categorical_column_index) is not list:
            categorical_column_index = [categorical_column_index]
        data = pd.read_csv(file, sep=column_separator, header=0, error_bad_lines=False)

        columns = data.columns
        # if (self.verbose):
        #     print(data)
        #     print(columns)
        #     print(categoricalColumnIndex)
        if self.verbose:
            print("before discretization: ")
            print("features: ", columns)
            print("index of categorical features: ", categorical_column_index)
        #
        # if (self.verbose):
        #     print(data.columns)

        column_Y = columns[-1]

        data.dropna(axis=1, thresh=frac_present * len(data), inplace=True)
        data.dropna(axis=0, how='any', inplace=True)

        y = data.pop(column_Y).copy()

        # Initialize dataframe and thresholds
        X = pd.DataFrame(columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))
        thresh = {}
        column_counter = 1
        self.column_info = []
        # Iterate over columns
        count = 0
        for c in data:
            # number of unique values
            val_uniq = data[c].nunique()

            # Constant column --- discard
            if val_uniq < 2:
                continue

            # Binary column
            elif val_uniq == 2:
                # Rename values to 0, 1
                X[('is', c, '')] = data[c].replace(np.sort(data[c].unique()), [0, 1])
                X[('is not', c, '')] = data[c].replace(np.sort(data[c].unique()), [1, 0])

                temp = [1, column_counter, column_counter + 1]
                self.column_info.append(temp)
                column_counter += 2

            # Categorical column
            elif (count in categorical_column_index) or (data[c].dtype == 'object'):
                # if (self.verbose):
                #     print(c)
                #     print(c in categoricalColumnIndex)
                #     print(data[c].dtype)
                # Dummy-code values
                Anew = pd.get_dummies(data[c]).astype(int)
                Anew.columns = Anew.columns.astype(str)
                # Append negations
                Anew = pd.concat([Anew, 1 - Anew], axis=1, keys=[(c, '=='), (c, '!=')])
                # Concatenate
                X = pd.concat([X, Anew], axis=1)

                addedColumn = len(Anew.columns)
                addedColumn = int(addedColumn / 2)
                temp = [2]
                temp = temp + [column_counter + nc for nc in range(addedColumn)]
                column_counter += addedColumn
                self.column_info.append(temp)
                temp = [3]
                temp = temp + [column_counter + nc for nc in range(addedColumn)]
                column_counter += addedColumn
                self.column_info.append(temp)

            # Ordinal column
            elif np.issubdtype(data[c].dtype, np.integer) | np.issubdtype(data[c].dtype, np.floating):
                # Few unique values
                # if (self.verbose):
                #     print(data[c].dtype)
                if val_uniq <= num_threshold + 1:
                    # Thresholds are sorted unique values excluding maximum
                    thresh[c] = np.sort(data[c].unique())[:-1]
                # Many unique values
                else:
                    # Thresholds are quantiles excluding repetitions
                    thresh[c] = data[c].quantile(q=quant_prob).unique()
                # Threshold values to produce binary arrays
                Anew = (data[c].values[:, np.newaxis] <= thresh[c]).astype(int)
                Anew = np.concatenate((Anew, 1 - Anew), axis=1)
                # Convert to dataframe with column labels
                Anew = pd.DataFrame(Anew,
                                    columns=pd.MultiIndex.from_product([[c], ['<=', '>'], thresh[c].astype(str)]))
                # Concatenate
                # print(A.shape)
                # print(Anew.shape)
                X = pd.concat([X, Anew], axis=1)

                addedColumn = len(Anew.columns)
                addedColumn = int(addedColumn / 2)
                temp = [4]
                temp = temp + [column_counter + nc for nc in range(addedColumn)]
                column_counter += addedColumn
                self.column_info.append(temp)
                temp = [5]
                temp = temp + [column_counter + nc for nc in range(addedColumn)]
                column_counter += addedColumn
                self.column_info.append(temp)
            else:
                # print(("Skipping column '" + c + "': data type cannot be handled"))
                continue
            count += 1
        self.columns = X.columns
        return X.to_numpy(), y.values.ravel()

    def learn_model(self, X, y, isTest):
        # temp files to save maxsat query in wcnf format
        WCNFFile = self.workDir + "/" + "model.wcnf"
        outputFileMaxsat = self.workDir + "/" + "model_out.txt"

        # generate maxsat query for dataset
        if (self.ruleType == 'DNF'):
            self.generateWCNFFile(X, y, len(X[0]),
                                  WCNFFile,
                                  isTest)

        else:
            #  negate yVector for CNF rules
            self.generateWCNFFile(X, [1 - int(y[each_y]) for each_y in
                                      range(len(y))],
                                  len(X[0]), WCNFFile,
                                  isTest)
        # call a maxsat solver
        if self.solver == "open-wbo":  # solver has timeout and experimented with open-wbo only
            if self.numPartition == -1:
                cmd = self.solver + '   ' + WCNFFile + ' -cpu-lim=' + str(self.timeOut) + ' > ' + outputFileMaxsat
            else:
                if int(math.ceil(self.timeOut / self.numPartition)) < 1:  # give at lest 1 second as cpu-lim
                    cmd = self.solver + '   ' + WCNFFile + ' -cpu-lim=' + str(1) + ' > ' + outputFileMaxsat
                else:
                    cmd = self.solver + '   ' + WCNFFile + ' -cpu-lim=' + str(
                        int(math.ceil(self.timeOut / self.numPartition))) + ' > ' + outputFileMaxsat
                    # print(int(math.ceil(self.timeOut/self.numPartition)))
        else:
            cmd = self.solver + '   ' + WCNFFile + ' > ' + outputFileMaxsat

        os.system(cmd)

        # delete temp files
        cmd = "rm " + WCNFFile
        os.system(cmd)

        # parse result of maxsat solving
        f = open(outputFileMaxsat, 'r')
        lines = f.readlines()
        f.close()
        optimumFound = False
        bestSolutionFound = False
        solution = ''
        for line in lines:
            if line.strip().startswith('v'):
                solution = line.strip().strip('v ')
                break

        fields = solution.split()
        TrueRules = []
        TrueErrors = []
        zeroOneSolution = []

        fields = self.prune_rules(fields, self.column_info[-1][-1])

        # Tirando possiveis variaveis postas pelo solver

        # Para cada regra j das N regras
        for j in range(self.numClause):

            # Para cada feature r das K features
            for r in range(len(self.column_info)):

                # Se a coluna for binaria
                if self.column_info[r][0] == 1:
                    try:
                        fields.remove(str(self.column_info[r][2] + (j * self.column_info[-1][-1])))
                    except:
                        fields.remove('-' + str(self.column_info[r][2] + (j * self.column_info[-1][-1])))

                # Se a coluna for categorica ou ordinal
                elif self.column_info[r][0] == 3 or self.column_info[r][0] == 5:
                    # Para cada subcoluna sc
                    for sc in range(1, len(self.column_info[r])):
                        try:
                            fields.remove(str(self.column_info[r][sc] + (j * self.column_info[-1][-1])))
                        except:
                            fields.remove('-' + str(self.column_info[r][sc] + (j * self.column_info[-1][-1])))

                # Se não for binaria, categoria ou ordinal e coluna barrada
                else:
                    continue

        for field in fields:
            if int(field) > 0:
                zeroOneSolution.append(1.0)

            else:
                zeroOneSolution.append(0.0)

                # Averiguando se esse literal representa uma coluna
                if abs(int(field)) <= self.numClause * self.column_info[-1][-1]:
                    TrueRules.append(str(abs(int(field))))

        self.xhat = []
        self.xhatField = []

        for i in range(self.numClause):
            self.xhat.append(
                np.concatenate((
                    np.array(
                        zeroOneSolution[i * (self.column_info[-1][-1] // 2):(i + 1) * (self.column_info[-1][-1] // 2)]),
                    np.array(zeroOneSolution[
                             i * (self.column_info[-1][-1] // 2) + self.numClause * (self.column_info[-1][-1] // 2):
                             ((i * (self.column_info[-1][-1] // 2)) + self.numClause * (self.column_info[-1][-1] // 2))
                             + (self.column_info[-1][-1] // 2)]
                             )
                )
                )
            )

            self.xhatField.append(
                np.concatenate((
                    np.array(fields[i * (self.column_info[-1][-1] // 2):(i + 1) * (self.column_info[-1][-1] // 2)]),
                    np.array(fields[
                             (i * (self.column_info[-1][-1] // 2)) + self.numClause * (self.column_info[-1][-1] // 2):((
                                                                                                                                 i * (
                                                                                                                                     self.column_info[
                                                                                                                                         -1][
                                                                                                                                         -1] // 2)) + self.numClause * (
                                                                                                                                 self.column_info[
                                                                                                                                     -1][
                                                                                                                                     -1] // 2)) + (
                                                                                                                                self.column_info[
                                                                                                                                    -1][
                                                                                                                                    -1] // 2)])
                ))
            )

        # delete temp files
        cmd = "rm " + outputFileMaxsat
        os.system(cmd)

        if not isTest:
            self.assignList = fields[:(self.numClause * (self.column_info[-1][-1] // 2)) + (
                        self.numClause * (self.column_info[-1][-1] // 2))]
            self.trainingError += len(TrueErrors)
            self.selectedFeatureIndex = TrueRules

        return fields[
               (self.numClause * (self.column_info[-1][-1] // 2)) + (self.numClause * (self.column_info[-1][-1] // 2)):]

    def fit(self, XTrain, yTrain):

        if self.numPartition == -1:
            self.numPartition = 2 ** math.floor(math.log2(len(XTrain) / 16))

            # print("partitions:" + str(self.numPartition))

            if self.numPartition < 1:
                self.numPartition = 1

        self.trainingError = 0
        self.trainingSize = len(XTrain)

        XTrains, yTrains = self.partition_with_equal_probability(XTrain, yTrain)

        self.assignList = []
        for each_partition in range(self.numPartition):
            self.learn_model(XTrains[each_partition], yTrains[each_partition], isTest=False)

    def partition_with_equal_probability(self, X, y):
        """
            Steps:
                1. seperate data based on class value
                2. partition each separate data into partition_count batches using test_train_split method with 50% part in each
                3. merge one separate batch from each class and save
            :param X:
            :param y:
            :param partition_count:
            :param location:
            :param file_name_header:
            :param column_set_list: uses for incremental approach
            :return:
        """
        partition_count = self.numPartition
        # y = y.values.ravel()
        max_y = int(y.max())
        min_y = int(y.min())

        X_list = [[] for i in range(max_y - min_y + 1)]
        y_list = [[] for i in range(max_y - min_y + 1)]
        level = int(math.log(partition_count, 2.0))

        # ao final do for abaixo y_list guardara duas listas, a primeira com previsoes 0 e a segunda com as previsoes 1
        # consequentemente o X_list guardara as duas listas com as respectivas posicoes de suas previsoes do y_list
        for i in range(len(y)):
            inserting_index = int(y[i])
            y_list[inserting_index - min_y].append(y[i])
            X_list[inserting_index - min_y].append(X[i])

        final_partition_X_train = [[] for i in range(partition_count)]
        final_partition_y_train = [[] for i in range(partition_count)]

        # o for abaixo percorrera separadamente o conjunto com previsoes 0 depois com previsoes 1 para
        # efetuar o embaralhamento(for interno)
        for each_class in range(len(X_list)):
            partition_list_X_train = [X_list[each_class]]
            partition_list_y_train = [y_list[each_class]]

            for i in range(level):
                for j in range(int(math.pow(2, i))):
                    A_train_1, A_train_2, y_train_1, y_train_2 = train_test_split(
                        partition_list_X_train[int(math.pow(2, i)) + j - 1],
                        partition_list_y_train[int(math.pow(2, i)) + j - 1],
                        test_size=0.5,
                        random_state=None)  # random state for keeping consistency between lp and maxsat approach
                    partition_list_X_train.append(A_train_1)
                    partition_list_X_train.append(A_train_2)
                    partition_list_y_train.append(y_train_1)
                    partition_list_y_train.append(y_train_2)

            partition_list_y_train = partition_list_y_train[partition_count - 1:]
            partition_list_X_train = partition_list_X_train[partition_count - 1:]

            for i in range(partition_count):
                final_partition_y_train[i] = final_partition_y_train[i] + partition_list_y_train[i]
                final_partition_X_train[i] = final_partition_X_train[i] + partition_list_X_train[i]

        return final_partition_X_train[:partition_count], final_partition_y_train[:partition_count]
    
    def prune_rules(self, fields, xSize):

        new_fileds = fields
        # vetor com as colunas barradas (caso o dado seja binário)
        # vetor com a ultima coluna dos dados normais  (caso o dado seja categorico ou ordinal)
        end_of_column_list = [self.column_info[i][-1] for i in range(len(self.column_info))]

        # matriz cuja linha representa uma regra(this.numClause) e cada coluna é um vetor que guarda freq. e
        # classificacao das respectivas colunas salvas anteriormente no end_of_column_list
        freq_end_of_column_list = [[[0, 0] for i in range(len(end_of_column_list))] for j in range(self.numClause)]

        # matriz que guarda os literais positivos de suas repectivas colunas barrada
        variable_contained_list = [[[] for i in range(len(end_of_column_list))] for j in range(self.numClause)]

        # variavel que representa o indice das variaveis l's no array fields
        l_position = self.numClause * self.column_info[-1][-1]

        # percorre todas as colunas que estarao na(s) regra(s)
        for i in range(self.numClause * xSize):
            # se o valor do literal nessa coluna for negativo (vai estar na regra) ...
            if int(fields[i]) < 0:
                # variavel que representa o valor do literal (nunca maior que o valor do literal que representa a
                # ultima coluna)
                variable = (abs(int(fields[i])) - 1) % xSize + 1

                # variavel que guarda o indice da regra que o literal esta (nunca maior que o numero de regras -1 (
                # indice))
                clause_position = int((abs(int(fields[i])) - 1) / xSize)

                # percorre todas as colunas adicionadas no end_of_column_list
                for j in range(len(end_of_column_list)):

                    # Averiguando se o valor do literal e menor ou igual ao valor da coluna guardada no indice j do
                    # end_of_column_list
                    if variable <= end_of_column_list[j]:

                        # Averiguando se a coluna e normal (binaria), pois se for eu pulo o lj,r dela
                        if self.column_info[j][0] == 1:
                            if variable == self.column_info[j][1]:
                                # Proxima coluna, entao proximo l
                                l_position += 1
                            break

                        # Averiguando se a coluna e normal (categorica), pois se for eu pulo o lj,r dela
                        elif self.column_info[j][0] == 2:
                            # Proxima coluna, entao proximo l
                            l_position += 1
                            break

                        # Averiguo se ela e ordinal normal
                        elif self.column_info[j][0] == 4:
                            variable_contained_list[clause_position][j].append(clause_position * xSize + variable)
                            freq_end_of_column_list[clause_position][j][0] += 1

                            # Averiguo se a polaridade dela e normal ou barrada
                            if int(fields[l_position]) > 0:
                                freq_end_of_column_list[clause_position][j][1] = self.column_info[j][0]
                            else:
                                freq_end_of_column_list[clause_position][j][1] = 5

                            # Proxima coluna, entao proximo l
                            l_position += 1

                            break

                        # Caso caia aqui, essa coluna nao precisa ser registrada, passo para a proxima
                        else:
                            break

            # Se for positivo, preciso averiguar se e alguma coluna normal para contar o indice do lj'r
            else:
                # variavel que representa o valor do literal (nunca maior que o valor do literal que representa a ultima coluna)
                variable = (int(fields[i]) - 1) % xSize + 1

                # percorre todas as colunas adicionadas no end_of_column_list
                for j in range(len(end_of_column_list)):

                    # Averiguando se o valor do literal e menor ou igual ao valor da coluna guardada no indice j do end_of_column_list
                    if variable <= end_of_column_list[j]:

                        # Averiguando se a coluna e normal (binaria), pois se for eu pulo o lj,r dela
                        if self.column_info[j][0] == 1:
                            if variable == self.column_info[j][1]:
                                # Proxima coluna, entao proximo l
                                l_position += 1
                            break

                        # Averiguando se a coluna e normal (binaria, categorica), pois se for eu pulo o lj,r dela
                        elif self.column_info[j][0] == 2 or self.column_info[j][0] == 4:
                            # Proxima coluna, entao proximo lj'r
                            l_position += 1
                            break

                        # Caso caia aqui, essa coluna nao precisa ser registrada, passo para a proxima
                        else:
                            break

        # percorrera o numero de regras
        for l in range(self.numClause):

            # percorre todas as colunas adicionadas no freq_end_of_column_list na linha l
            for i in range(len(freq_end_of_column_list[l])):
                # se a coluna apareceu mais de uma vez (freq. > 1) ...
                if freq_end_of_column_list[l][i][0] > 1:
                    # se a coluna for do tipo 4 (ordinal normal)
                    if freq_end_of_column_list[l][i][1] == 4:
                        # retiro o primeiro literal da lista de literais que representam a mesma coluna (ele vai ser o unico negativo ao final desse if)
                        variable_contained_list[l][i] = variable_contained_list[l][i][1:]
                        # retiro o sinal de negacao de todos os literais que ficaram na lista
                        for j in range(len(variable_contained_list[l][i])):
                            new_fileds[variable_contained_list[l][i][j] - 1] = str(
                                variable_contained_list[l][i][j])
                    # se a coluna for do tipo 5 (ordinal barrada)
                    elif freq_end_of_column_list[l][i][1] == 5:
                        # retiro o ultimo literal da lista de literais que representam a mesma coluna (ele vai ser o unico negativo ao final desse if)
                        variable_contained_list[l][i] = variable_contained_list[l][i][:-1]
                        for j in range(len(variable_contained_list[l][i])):
                            new_fileds[variable_contained_list[l][i][j] - 1] = str(
                                variable_contained_list[l][i][j])
        return new_fileds

    def variables_encoding(self, y_vector):
        index_var = 1
        variables_encoding = {}

        # x_ij,feature
        for clause_i in range(1, self.num_rules + 1):
            for literal_j in range(1, self.max_rule_size + 1):
                variables_encoding[('p', clause_i, literal_j)] = index_var  # polarity
                index_var = index_var + 1
                for feature in self.column_info:
                    variables_encoding[('x', clause_i, literal_j, feature)] = index_var
                    index_var = index_var + 1

        # x_ij,none
        for clause_i in range(1, self.num_rules + 1):
            for literal_j in range(1, self.max_rule_size + 1):
                variables_encoding[('x', clause_i, literal_j, 'none')] = index_var
                index_var = index_var + 1

        # truth value variables
        for example in range(1, len(y_vector) + 1):
            for clause_i in range(1, self.num_rules + 1):
                variables_encoding[('y', clause_i, example)] = index_var
                index_var = index_var + 1
                for literal_j in range(1, self.max_rule_size + 1):
                    variables_encoding[('y', clause_i, literal_j, example)] = index_var
                    index_var = index_var + 1

        # # dealing with noise
        # # f_w
        # for w in pos + neg:
        #     variables_encoding[('f', w)] = index_var
        #     index_var = index_var + 1
        #
        # # a_i,w and o_i,w
        # for i in range(1, number_noise + 1):
        #     for w in pos + neg:
        #         variables_encoding[('a', i, w)] = index_var
        #         index_var = index_var + 1
        #         variables_encoding[('o', i, w)] = index_var
        #         index_var = index_var + 1

        return pd.Series(variables_encoding)


# instancio o modelo indicando o nome do exec. do solver que deve está na mesma pasta
model = MaxSizeModel(solver="/home/thiago/PycharmProjects/TCC-Algoritmos/UWrMaxSat-1.1w/bin/uwrmaxsat")

# guardo o endereco da tabela que será usada para a aplicacao do modelo (... -> end. da pasta do projeto)
# arq = r"/home/thiago/PycharmProjects/TCC-Algoritmos/Tabela_de_testes/parkinsons.data.csv"
arq = r"/home/thiago/PycharmProjects/TCC-Algoritmos/Tabela_de_testes/column_2C.csv"

# aplico a discretizacao do modelo na tabela
X, y = model.discretize(arq)

print(X.shape)
print(y.shape)

print(model.variables_encoding(y))



