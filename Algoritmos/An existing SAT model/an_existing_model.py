import numpy as np
import pandas as pd
import warnings
import math
import os
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)


class an_existing_model():
    def __init__(self, numPartition=-1, numClause=2, dataFidelity=10, weightFeature=1, solver="open-wbo",
                 ruleType="DNF",
                 workDir=".", timeOut=1024):
        '''

        :param numPartition: no of partitions of training dataset
        :param numClause: no of clause in the formula
        :param dataFidelity: weight corresponding to accuracy
        :param weightFeature: weight corresponding to selected features
        :param solver: specify the (name of the) bin of the solver; bin must be in the path
        :param ruleType: type of rule {CNF,DNF}
        :param workDir: working directory
        :param verbose: True for debug
        '''
        self.numPartition = numPartition
        self.numClause = numClause
        self.dataFidelity = dataFidelity
        self.weightFeature = weightFeature
        self.solver = solver
        self.ruleType = ruleType
        self.workDir = workDir
        self.verbose = False  # not necessary
        self.trainingError = 0
        self.selectedFeatureIndex = []
        self.columns = []
        self.timeOut = timeOut

    def __repr__(self):
        return "imli: -->  \nnumPartition:%s \nnumClause:%s \ndataFidelity:%s \nweightFeature:%s " \
               "\nsolver:%s \nruleType:%s \nworkDir:%s \ntimeout:%s>" % (self.numPartition,
                                                                         self.numClause, self.dataFidelity,
                                                                         self.weightFeature, self.solver,
                                                                         self.ruleType, self.workDir, self.timeOut)

    def getColumns(self):
        return [str(a) + str(" ") + str(b) + str(" ") + str(c) for (a, b, c) in self.columns]

    def getTrainingError(self):
        return self.trainingError

    def getSelectedColumnIndex(self):
        return_list = [[] for i in range(self.numClause)]
        ySize = len(self.columns)
        for elem in self.selectedFeatureIndex:
            new_index = int(elem) - 1
            return_list[int(new_index / ySize)].append(new_index % ySize)
        return return_list

    def getNumOfPartition(self):
        return self.numPartition

    def getNumOfClause(self):
        return self.numClause

    def getWeightFeature(self):
        return self.weightFeature

    def getRuleType(self):
        return self.ruleType

    def getWorkDir(self):
        return self.workDir

    def getWeightDataFidelity(self):
        return self.dataFidelity

    def getSolver(self):
        return self.solver

    def getRuleSize(self):
        ruleIndex = self.getSelectedColumnIndex()
        sR = 0
        for j in range(len(ruleIndex)):
            sR += len(ruleIndex[j])

        return sR

    def getBiggestRuleSize(self):
        ruleIndex = self.getSelectedColumnIndex()
        bR = 0
        for j in range(len(ruleIndex)):
            if (len(ruleIndex[j]) >= bR):
                bR = len(ruleIndex[j])

        return bR

    def discretize(self, file, categoricalColumnIndex=[], columnSeperator=",", fracPresent=0.9, numThreshold=4):

        # Quantile probabilities
        quantProb = np.linspace(1. / (numThreshold + 1.), numThreshold / (numThreshold + 1.), numThreshold)
        # List of categorical columns
        if type(categoricalColumnIndex) is pd.Series:
            categoricalColumnIndex = categoricalColumnIndex.tolist()
        elif type(categoricalColumnIndex) is not list:
            categoricalColumnIndex = [categoricalColumnIndex]
        data = pd.read_csv(file, sep=columnSeperator, header=0, error_bad_lines=False)

        columns = data.columns
        # if (self.verbose):
        #     print(data)
        #     print(columns)
        #     print(categoricalColumnIndex)
        if (self.verbose):
            print("before discrertization: ")
            print("features: ", columns)
            print("index of categorical features: ", categoricalColumnIndex)
        #
        # if (self.verbose):
        #     print(data.columns)

        columnY = columns[-1]

        data.dropna(axis=1, thresh=fracPresent * len(data), inplace=True)
        data.dropna(axis=0, how='any', inplace=True)

        y = data.pop(columnY).copy()

        # Initialize dataframe and thresholds
        X = pd.DataFrame(columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))
        thresh = {}
        column_counter = 1
        self.columnInfo = []
        # Iterate over columns
        count = 0
        for c in data:
            # number of unique values
            valUniq = data[c].nunique()

            # Constant column --- discard
            if valUniq < 2:
                continue

            # Binary column
            elif valUniq == 2:
                # Rename values to 0, 1
                X[('is', c, '')] = data[c].replace(np.sort(data[c].unique()), [0, 1])
                X[('is not', c, '')] = data[c].replace(np.sort(data[c].unique()), [1, 0])

                temp = [1, column_counter, column_counter + 1]
                self.columnInfo.append(temp)
                column_counter += 2

            # Categorical column
            elif (count in categoricalColumnIndex) or (data[c].dtype == 'object'):
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
                self.columnInfo.append(temp)
                temp = [3]
                temp = temp + [column_counter + nc for nc in range(addedColumn)]
                column_counter += addedColumn
                self.columnInfo.append(temp)

            # Ordinal column
            elif np.issubdtype(data[c].dtype, np.integer) | np.issubdtype(data[c].dtype, np.floating):
                # Few unique values
                # if (self.verbose):
                #     print(data[c].dtype)
                if valUniq <= numThreshold + 1:
                    # Thresholds are sorted unique values excluding maximum
                    thresh[c] = np.sort(data[c].unique())[:-1]
                # Many unique values
                else:
                    # Thresholds are quantiles excluding repetitions
                    thresh[c] = data[c].quantile(q=quantProb).unique()
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
                self.columnInfo.append(temp)
                temp = [5]
                temp = temp + [column_counter + nc for nc in range(addedColumn)]
                column_counter += addedColumn
                self.columnInfo.append(temp)
            else:
                # print(("Skipping column '" + c + "': data type cannot be handled"))
                continue
            count += 1
        self.columns = X.columns
        return X.to_numpy(), y.values.ravel()

    def fit(self, XTrain, yTrain):

        if (self.numPartition == -1):
            self.numPartition = 2 ** math.floor(math.log2(len(XTrain) / 16))

            # print("partitions:" + str(self.numPartition))

            if (self.numPartition < 1):
                self.numPartition = 1

        self.trainingError = 0
        self.trainingSize = len(XTrain)

        XTrains, yTrains = self.partitionWithEqualProbability(XTrain, yTrain)

        self.assignList = []
        for each_partition in range(self.numPartition):
            self.learnModel(XTrains[each_partition], yTrains[each_partition], isTest=False)

    def predict(self, XTest):
        rule = self.getSelectedColumnIndex()
        y = []

        prediction = 0

        # para cada linha da matriz
        for e in range(len(XTest)):

            # para cada clausula
            for j in range(len(rule)):

                # para cada coluna da clausula
                for r in range(len(rule[j])):
                    if (self.ruleType == 'DNF'):
                        if (XTest[e][rule[j][r]] == 0):
                            prediction = 0
                            break
                        else:
                            prediction = 1
                    elif (self.ruleType == 'CNF'):
                        if (XTest[e][rule[j][r]] == 1):
                            prediction = 1
                            break
                        else:
                            prediction = 0

                if (self.ruleType == 'DNF' and prediction == 1):
                    break
                elif (self.ruleType == 'CNF' and prediction == 0):
                    break

            y.append(prediction)

        return y

    def score(self, XTest, y):
        yTest = self.predict(XTest)

        hits = 0
        for i in range(len(yTest)):
            if (yTest[i] == y[i]):
                hits += 1

        return hits / len(yTest)

    def learnModel(self, X, y, isTest):
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
        if (self.solver == "open-wbo"):  # solver has timeout and experimented with open-wbo only
            if (self.numPartition == -1):
                cmd = self.solver + '   ' + WCNFFile + ' -cpu-lim=' + str(self.timeOut) + ' > ' + outputFileMaxsat
            else:
                if (int(math.ceil(self.timeOut / self.numPartition)) < 1):  # give at lest 1 second as cpu-lim
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
            if (line.strip().startswith('v')):
                solution = line.strip().strip('v ')
                break

        fields = solution.split()
        TrueRules = []
        TrueErrors = []
        zeroOneSolution = []

        fields = self.pruneRules(fields, self.columnInfo[-1][-1])

        for field in fields:
            if (int(field) > 0):
                zeroOneSolution.append(1.0)

            else:
                zeroOneSolution.append(0.0)

                # Averiguando se esse literal representa uma coluna
                if (abs(int(field)) <= self.numClause * self.columnInfo[-1][-1]):
                    TrueRules.append(str(abs(int(field))))

        self.xhat = []

        for i in range(self.numClause):
            self.xhat.append(np.array(zeroOneSolution[i * self.columnInfo[-1][-1]:(i + 1) * self.columnInfo[-1][-1]]))

        # delete temp files
        cmd = "rm " + outputFileMaxsat
        os.system(cmd)

        if (not isTest):
            self.assignList = fields[:self.numClause * (self.columnInfo[-1][-1])]
            self.trainingError += len(TrueErrors)
            self.selectedFeatureIndex = TrueRules

        return fields[self.numClause * self.columnInfo[-1][-1]:]

    def partitionWithEqualProbability(self, X, y):
        '''
            Steps:
                1. seperate data based on class value
                2. partition each seperate data into partition_count batches using test_train_split method with 50% part in each
                3. merge one seperate batche from each class and save
            :param X:
            :param y:
            :param partition_count:
            :param location:
            :param file_name_header:
            :param column_set_list: uses for incremental approach
            :return:
            '''
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

    def generateWCNFFile(self, AMatrix, yVector, xSize, WCNFFile,
                         isTestPhase):
        ''' VARIÁVEIS IMPORTANTES

        self.numClause = representa o número de regras que se quer obter

        additionalVariable = representa o número de literais total do arquivo WCNF
        numClauses = representa o número de cláusulas geradas pela modelagem (serve para ser inserida no arq. com o padrão DIMAC)
        topWeight = representa o número maior dentre os pesos (serve para ser inserida no arq. com o padrão DIMAC)

        cnfClauses = representa o arquivo WCNF gerado ao final da modelagem

        '''
        topWeight = 1
        additionalVariable = 0
        numClauses = 0
        cnfClauses = ''

        # MONTANDO CLAUSULAS SOFT, ---------------------------------------------------------------------------------------------------

        # 1.1) Criando variaveis soft que garantem o minimo de colunas contida nas regras

        # Se o assignList estiver vazio, entao quer dizer que estamos na primeira particao de dados,
        # logo devemos CRIAR os literais que representam as colunas e sua polaridade
        if (self.assignList == []):

            # Para cada regra j das N regras
            for j in range(self.numClause):

                # Para cada feature r das K features
                for r in range(len(self.columnInfo)):

                    # Se a coluna for binaria
                    if (self.columnInfo[r][0] == 1):
                        new_clause = str(topWeight) + ' '

                        # Para a criacao do literal: pj,r
                        new_clause += str(self.columnInfo[r][1] + (j * self.columnInfo[-1][-1]))
                        new_clause += " 0\n"
                        cnfClauses += new_clause

                        new_clause = str(topWeight) + ' '

                        # Para a criacao do literal: p'j,r
                        new_clause += str(self.columnInfo[r][2] + (j * self.columnInfo[-1][-1]))
                        new_clause += " 0\n"
                        cnfClauses += new_clause

                        additionalVariable += 2
                        numClauses += 2


                    # Se a coluna for categorica ou ordinal
                    elif (self.columnInfo[r][0] == 2 or self.columnInfo[r][0] == 4):
                        # Para cada subcoluna sc
                        for sc in range(1, len(self.columnInfo[r])):
                            new_clause = str(topWeight) + ' '

                            # Para a criacao de dois literais: pj,r
                            new_clause += str(self.columnInfo[r][sc] + (j * self.columnInfo[-1][-1]))
                            new_clause += " 0\n"
                            cnfClauses += new_clause

                            new_clause = str(topWeight) + ' '

                            # Para a criacao de dois literais: p'j,r
                            new_clause += str(self.columnInfo[r + 1][sc] + (j * self.columnInfo[-1][-1]))
                            new_clause += " 0\n"
                            cnfClauses += new_clause

                            additionalVariable += 2
                            numClauses += 2


                    # Se não for binaria, categoria ou ordinal e coluna barrada
                    else:
                        continue

        # Caso o assignList esteja com valores, devemos usa-los
        else:
            # Para cada literal no assignList
            for l in self.assignList:
                new_clause = str(topWeight) + ' '

                new_clause += str(l)
                new_clause += " 0\n"
                cnfClauses += new_clause

                additionalVariable += 1
                numClauses += 1

        # Atualizando o topWeight
        topWeight = self.numClause * self.columnInfo[-1][-1]

        # 2.1) Garante que uma linha com y = 0 seja falsa em todas as regras.

        # Para cada linha da matriz de dados ...
        for e in range(len(yVector)):

            # Se essa linha tiver previsao = 0
            if (yVector[e] == 0):

                # Para cada regra j das N regras
                for j in range(self.numClause):
                    new_clause = str(self.dataFidelity) + ' '
                    topWeight += self.dataFidelity

                    # Para cada feature r das K features da linha e
                    for r in range(len(self.columnInfo)):

                        # Se a coluna for binaria
                        if (self.columnInfo[r][0] == 1):

                            # Se o valor desse dessa linha e na coluna r for == 1
                            if (AMatrix[e][self.columnInfo[r][1] - 1] == 1):
                                new_clause += '-' + str(self.columnInfo[r][2] + (j * self.columnInfo[-1][-1])) + ' '
                            else:
                                new_clause += '-' + str(self.columnInfo[r][1] + (j * self.columnInfo[-1][-1])) + ' '

                        # Se a coluna for categorica ou ordinal
                        elif (self.columnInfo[r][0] == 2 or self.columnInfo[r][0] == 4):

                            # Para cada subcoluna sc
                            for sc in range(1, len(self.columnInfo[r])):

                                # Se o valor desse dessa linha e na coluna r(sc) for == 1
                                if (AMatrix[e][self.columnInfo[r][sc] - 1] == 1):
                                    new_clause += '-' + str(
                                        self.columnInfo[r + 1][sc] + (j * self.columnInfo[-1][-1])) + ' '
                                else:
                                    new_clause += '-' + str(
                                        self.columnInfo[r][sc] + (j * self.columnInfo[-1][-1])) + ' '

                        # Se não for binaria, categoria ou ordinal, entao -> coluna barrada
                        else:
                            continue

                    new_clause += "0\n"
                    numClauses += 1
                    cnfClauses += new_clause

        # 4.1 ) Garante que a linha com y = 1 tem que ser verdadeira em alguma regra.

        # Percorrendo as variaveis cr,j
        # OBS: Como as variáveis cr,j ainda não foram criadas, precisamos prever quantas terão no total
        additionalVariablePrevision = additionalVariable + self.numClause * yVector.count(1)

        for cr in range((self.numClause * self.columnInfo[-1][-1]) + 1, additionalVariablePrevision + 1,
                        self.numClause):
            new_clause = str(self.dataFidelity) + ' '
            topWeight += self.dataFidelity

            # Para cada regra j das N regras
            for j in range(self.numClause):
                new_clause += str(cr) + ' '

                cr += 1

            new_clause += "0\n"
            numClauses += 1
            cnfClauses += new_clause

        # Fazendo com que o topWeight seja maior que a soma das softClauses para pesificar as hard's
        topWeight += 1

        # MONTANDO CLAUSULAS HARD, ---------------------------------------------------------------------------------------------------

        # 1) Garantindo que duas colunas não estejam ao mesmo tempo com polaridades invertidas. Ex: A ou ¬A, nunca os dois.

        # Para cada regra j das N regras
        for j in range(self.numClause):

            # Para cada feature r das K features
            for r in range(len(self.columnInfo)):

                # Se a coluna for binaria
                if (self.columnInfo[r][0] == 1):
                    new_clause = str(topWeight) + ' '

                    # Para a criacao da clausula: (pj,r V p'j,r)
                    new_clause += str(self.columnInfo[r][1] + (j * self.columnInfo[-1][-1])) + ' ' + str(
                        self.columnInfo[r][2] + (j * self.columnInfo[-1][-1]))

                    new_clause += " 0\n"
                    numClauses += 1
                    cnfClauses += new_clause

                # Se a coluna for categorica ou ordinal
                elif (self.columnInfo[r][0] == 2 or self.columnInfo[r][0] == 4):
                    # Para cada subcoluna sc
                    for sc in range(1, len(self.columnInfo[r])):
                        new_clause = str(topWeight) + ' '

                        # Para a criacao da clausula: (pj,r V p'j,r)
                        new_clause += str(self.columnInfo[r][sc] + (j * self.columnInfo[-1][-1])) + ' ' + str(
                            self.columnInfo[r + 1][sc] + (j * self.columnInfo[-1][-1]))

                        new_clause += " 0\n"
                        numClauses += 1
                        cnfClauses += new_clause

                # Se não for binaria, categoria ou ordinal e coluna barrada
                else:
                    continue

        # 3) ...
        for e in range(len(yVector)):

            # Se essa linha tiver previsao = 1
            if (yVector[e] == 1):

                # Para cada regra j das N regras
                for j in range(self.numClause):
                    # Representa a criacao de uma variável cr,j
                    additionalVariable += 1

                    # Para cada feature r das K features da linha e
                    for r in range(len(self.columnInfo)):
                        new_clause = str(topWeight) + ' '

                        # Se a coluna for binaria
                        if (self.columnInfo[r][0] == 1):

                            # Se o valor desse dessa linha e na coluna r for == 1
                            if (AMatrix[e][self.columnInfo[r][1] - 1] == 1):
                                new_clause += str(self.columnInfo[r][2] + (j * self.columnInfo[-1][-1])) + ' -' + str(
                                    additionalVariable)
                            else:
                                new_clause += str(self.columnInfo[r][1] + (j * self.columnInfo[-1][-1])) + ' -' + str(
                                    additionalVariable)

                            new_clause += " 0\n"
                            numClauses += 1
                            cnfClauses += new_clause

                        # Se a coluna for categorica ou ordinal
                        elif (self.columnInfo[r][0] == 2 or self.columnInfo[r][0] == 4):

                            # Para cada subcoluna sc
                            for sc in range(1, len(self.columnInfo[r])):
                                new_clause = str(topWeight) + ' '

                                # Se o valor desse dessa linha e na coluna r(sc) for == 1
                                if (AMatrix[e][self.columnInfo[r][sc] - 1] == 1):
                                    new_clause += str(
                                        self.columnInfo[r + 1][sc] + (j * self.columnInfo[-1][-1])) + ' -' + str(
                                        additionalVariable)
                                else:
                                    new_clause += str(
                                        self.columnInfo[r][sc] + (j * self.columnInfo[-1][-1])) + ' -' + str(
                                        additionalVariable)

                                new_clause += " 0\n"
                                numClauses += 1
                                cnfClauses += new_clause

                        # Se não for binaria, categoria ou ordinal, entao -> coluna barrada
                        else:
                            continue

        # write in wcnf format
        header = 'p wcnf ' + str(additionalVariable) + ' ' + str(numClauses) + ' ' + str(topWeight) + '\n'
        f = open(WCNFFile, 'w')
        f.write(header)
        f.write(cnfClauses)
        f.close()

    def pruneRules(self, fields, xSize):

        new_fileds = fields
        # vetor com as colunas barradas (caso o dado seja binário)
        # vetor com a ultima coluna dos dados normais  (caso o dado seja categorico ou ordinal)
        end_of_column_list = [self.columnInfo[i][-1] for i in range(len(self.columnInfo))]

        # matriz cuja linha representa uma regra(this.numClause) e cada coluna é um vetor que guarda freq. e
        # classificacao das respectivas colunas salvas anteriormente no end_of_column_list
        freq_end_of_column_list = [[[0, 0] for i in range(len(end_of_column_list))] for j in range(self.numClause)]

        # matriz que guarda os literais positivos de suas repectivas colunas barrada
        variable_contained_list = [[[] for i in range(len(end_of_column_list))] for j in range(self.numClause)]

        # percorre todas as colunas que estarao na(s) regra(s)
        for i in range(self.numClause * xSize):
            # se o valor do literal nessa coluna for negativo (vai estar na regra) ...
            if ((int(fields[i])) < 0):
                # variavel que representa o valor do literal (nunca maior que o valor do literal que representa a ultima coluna)
                variable = (abs(int(fields[i])) - 1) % xSize + 1

                # variavel que guarda o indice da regra que o literal esta (nunca maior que o numero de regras -1 (indice))
                clause_position = int((abs(int(fields[i])) - 1) / xSize)

                # percorre todas as colunas adicionadas no end_of_column_list
                for j in range(len(end_of_column_list)):

                    # averiguando se o valor do literal e menor ou igual ao valor da coluna guardada no indice j do end_of_column_list
                    if (variable <= end_of_column_list[j]):
                        variable_contained_list[clause_position][j].append(clause_position * xSize + variable)
                        freq_end_of_column_list[clause_position][j][0] += 1
                        freq_end_of_column_list[clause_position][j][1] = self.columnInfo[j][0]
                        break

        # percorrera o numero de regras
        for l in range(self.numClause):

            # percorre todas as colunas adicionadas no freq_end_of_column_list na linha l
            for i in range(len(freq_end_of_column_list[l])):
                # se a coluna apareceu mais de uma vez (freq. > 1) ...
                if (freq_end_of_column_list[l][i][0] > 1):
                    # se a coluna for do tipo 4 (ordinal normal)
                    if (freq_end_of_column_list[l][i][1] == 4):
                        # retiro o primeiro literal da lista de literais que representam a mesma coluna (ele vai ser o unico negativo ao final desse if)
                        variable_contained_list[l][i] = variable_contained_list[l][i][1:]
                        # retiro o sinal de negacao de todos os literais que ficaram na lista
                        for j in range(len(variable_contained_list[l][i])):
                            new_fileds[variable_contained_list[l][i][j] - 1] = str(
                                variable_contained_list[l][i][j])
                    # se a coluna for do tipo 5 (ordinal barrada)
                    elif (freq_end_of_column_list[l][i][1] == 5):
                        # retiro o ultimo literal da lista de literais que representam a mesma coluna (ele vai ser o unico negativo ao final desse if)
                        variable_contained_list[l][i] = variable_contained_list[l][i][:-1]
                        for j in range(len(variable_contained_list[l][i])):
                            new_fileds[variable_contained_list[l][i][j] - 1] = str(
                                variable_contained_list[l][i][j])
        return new_fileds

    def getRule(self):
        generatedRule = '( '
        for i in range(self.numClause):
            xHatElem = self.xhat[i]
            inds_nnz = np.where(abs(xHatElem) < 1e-4)[0]

            str_clauses = [' '.join(self.columns[ind]) for ind in inds_nnz]
            if (self.ruleType == "CNF"):
                rule_sep = ' %s ' % "or"
            else:
                rule_sep = ' %s ' % "and"
            rule_str = rule_sep.join(str_clauses)

            generatedRule += rule_str
            if (i < self.numClause - 1):
                if (self.ruleType == "DNF"):
                    generatedRule += ' ) or \n( '
                if (self.ruleType == 'CNF'):
                    generatedRule += ' ) and \n( '
        generatedRule += ')'

        return generatedRule


# ------------- TESTES --------------------------------------------------------------------

# instancio o modelo indicando o nome do exec. do solver que deve está na mesma pasta
model = an_existing_model(solver="/home/thiago/PycharmProjects/TCC-Algoritmos/UWrMaxSat-1.1w/bin/uwrmaxsat")

# guardo o endereco da tabela que será usada para a aplicacao do modelo (... -> end. da pasta do projeto)
#arq = r"/home/thiago/PycharmProjects/TCC-Algoritmos/Tabela_de_testes/parkinsons.data.csv"
#arq = r"/home/thiago/PycharmProjects/TCC-Algoritmos/Tabela_de_testes/column_bin.csv"
arq = r"/home/thiago/PycharmProjects/TCC-Algoritmos/Tabela_de_testes/column_2C.csv"

# aplico a discretizacao do modelo na tabela
X, y = model.discretize(arq)

# treinando o modelo usando a discretizacao da tabela
model.fit(X, y)

# guardando as regras geradas pelo treino
rule = model.getRule()
print('============== RULE ==============')
print(rule)
print('==================================')

'''
#indice das colunas que estao na regra
columnsError = model.getSelectedColumnIndex()
print('======= RULES COLUMN INDEX =======')
print(columnsError)
print('==================================')

#previsao do teste
print(model.predict(X))
'''

# precisao do teste
print('Score:', model.score(X, y))

# printando a quantidade de regras e a quantidade da maior, respectivamente
print('Rules number:', model.getRuleSize())
print('Biggest rule number:', model.getBiggestRuleSize())
