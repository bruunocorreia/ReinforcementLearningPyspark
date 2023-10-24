import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from scipy.stats import beta
from math import sqrt
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.functions import udf

class Mab:
    
    def __init__(self,
                 priors = None,
                 variable_test = None,
                 reward = None,
                 log_results = True):
        
        self._priors = priors
        self._variable_test = variable_test
        self._priors = priors
        self._reward = reward
        self._log_results = log_results
        
        #Validate number of experiments
        if self._variable_test is None:
            raise ValueError("Define the number of features")
        
        #Validate number of experiments
        
        if self._priors is not None:
            if self._priors.select('teste').agg(F.countDistinct('teste')).collect()[0][0] != len(variable_test):
                    raise ValueError("Define the number of features")
                    
        #Validate number of experiments
        if self._reward is None:
            raise ValueError("Define your reward metric")
            
        
    #Create epsilon greedy approach
    def EpsilonGreedy(self, epsilon):
        """ Function receive a spark or pandas DataFrame and define the next option using Epsilon Greedy approach
        
        Parameters:
            priors (Dataframe): DataFrame com os resultados anteriores
            variable_test (list): Lista de variaveis a serem testadas
            Epsilon (float): Valor entre 0 e 1 que a probabilidade de escolhas randomicas para o EpsilonGreedy
        Returns:
            Action to take using the previous data/results
        Authors:
            Bruno Correia """
    
        #Validate epsilon
        if epsilon is None:
            raise ValueError("Define your epsilon parameter")
        
        #Epsilon greedy
        random_number = random.uniform(0,1)
        
        #Check if you have priors
        if self._priors != None:
            
            #Exploration
            if random_number < epsilon:
                
                new_test = random.choice(self._variable_test)
                
                if  self._log_results == True:
                    print("Results")
                    print("Exploration - The best next feature to choose is", new_test)
                    print(f"(P = { random_number }, Epsilon = {epsilon}")
                    print("Actual distribution")
                    
                    #Agreggate data and metrics if necessary
                    df_agg_conversoes = (self._priors
                                            .groupby('teste')
                                            .agg(F.countDistinct('cpf').alias('qtd_cpfs'),
                                                F.sum('qtd_contatacoes').alias('qtd_contatacoes'))
                                            .withColumn('conversao', F.col('qtd_contatacoes')/F.col('qtd_cpfs'))
                                            .select('teste','conversao'))
                    df_agg_conversoes.show()
                    
            #Exploitation
            else:
                
                #Agreggate data and metrics if necessary
                df_agg_conversoes = (self._priors
                                            .groupby('teste')
                                            .agg(F.countDistinct('cpf').alias('qtd_cpfs'),
                                                F.sum(self._reward).alias('qtd_contatacoes'))
                                            .withColumn('conversao', F.col('qtd_contatacoes')/F.col('qtd_cpfs'))
                                            .select('teste','conversao'))
                
                #Select group with the highest conversion at this moment
                max_conversao = df_agg_conversoes.agg(F.max('conversao').alias('conversao'))
                
                new_test = (df_agg_conversoes
                                            .join(max_conversao, on = 'conversao', how = 'inner')
                                            .select('teste')
                                            .collect()[0][0])
                if  self._log_results == True:
                    print("Results")
                    print("Exploration - The best next feature to choose is", new_test)
                    print(f"(P = { random_number }, Epsilon = {epsilon}")
                    print("Actual distribution")
                    df_agg_conversoes.show()
                    
        #if you dont have priors we select a random option
        else:
            #Random selection
            proba_iguais = 1/len(self._variable_test)
            new_test = random.choice(self._variable_test)
            if  self._log_results == True:
                print(f"(P = { random_number }, Epsilon = {epsilon}")
                print("Exploration - The best next feature to choose is", new_test)
                print("We dont define priors, so we choose a random option")
        
        return new_test
    
    def plot_reward_epsilon_greedy(self):
        """ Plot distribution of the test"""
        plot_pandas = (self._priors
                            .groupby('dat_ref_carga')
                            .agg(F.sum(self._reward).alias('reward'))
                            .selectExpr(
                                        "dat_ref_carga",
                                        "reward",
                                        "sum(reward) over (order by row_number() over (order by reward desc)) as cumulative_reward"
                                        )
                            .toPandas())
        plt.plot(plot_pandas.dat_ref_carga,
                    plot_pandas.cumulative_reward)
        plt.title(f"Bandit distributions after trials")

        return plt.show()
    
    def Ucb(self):
        """ Function receive a spark or pandas DataFrame and define the next option using upper confidence bound approach
        
        Parameters:
            priors (Dataframe): DataFrame com os resultados anteriores
            variable_test (list): Lista de variaveis a serem testadas
        Returns:
            Action to take using the previous data/results
        Authors:
            Bruno Correia """
        
        #Check if you have priors
        if self._priors != None:
            total_rounds = self._priors.count()
            #Calculate bounds
            df_agg_ucb = (self._priors
                                .groupby('teste')
                                .agg(F.countDistinct('cpf').alias('qtd_cpfs'),
                                    F.sum(self._reward).alias('qtd_contratacoes'))
                                .withColumn('conversao', F.col('qtd_contratacoes')/F.col('qtd_cpfs'))
                                .withColumn('total_rounds', F.lit(total_rounds))
                                .withColumn('ucb', F.col('conversao') 
                                            + F.sqrt(2*F.log((((F.col('total_rounds'))/F.col('qtd_cpfs'))))))
                                
                            )
            #Select max bandit
            max_ucb = df_agg_ucb.agg(F.max('ucb').alias('ucb'))
            new_test = df_agg_ucb.join(max_ucb, on ='ucb',how = 'inner').select('teste').collect()[0][0]
            
            if  self._log_results == True:
                    print("Results")
                    print("The best next feature to choose is", new_test)
                    print("Actual distribution")
                    df_agg_ucb.select('teste','conversao','total_rounds','qtd_cpfs','ucb').show()
        else:
            #Random selection
            proba_iguais = 1/len(self._variable_test)
            new_test = random.choice(self._variable_test)
            if  self._log_results == True:
                print("Exploration - The best next feature to choose is", new_test)
                print("We dont define priors, so we choose a random option")
        
        return new_test
    
    @staticmethod
    @udf('double')
    def apply_random_beta(alfa,beta):
        value = np.random.beta(a=alfa,b=beta)
        return value
    
    def ThompsonSampling(self):
        """ Function receive a spark or pandas DataFrame and define the next option using Thompson Sampling/Bayesian Ab test
        
        Parameters:
            priors (Dataframe): DataFrame com os resultados anteriores
            variable_test (list): Lista de variaveis a serem testadas
        Returns:
            Action to take using the previous data/results
        Authors:
            Bruno Correia """
        

        #Check if you have priors
        if self._priors != None:
            #measure alfa and beta value for prior
            alfas_betas = (self._priors
                            .groupby('teste')
                            .agg(F.countDistinct(F.col('cpf')).alias('qtd'),
                                F.sum(F.col(self._reward)).alias('alfa'))
                            .withColumn('beta', F.col('qtd')-F.col('alfa'))
                            .orderBy('teste'))
            
            #calculate posterior distribution based on sampling using beta
            alfas_betas = (alfas_betas
                            .withColumn('exp_prob',
                                        self.apply_random_beta(alfas_betas.alfa,
                                                               alfas_betas.beta))).cache()
            
            #select max proba
            max_proba = alfas_betas.agg(F.max('exp_prob').alias('exp_prob'))
            new_test = (alfas_betas.join(max_proba, on = 'exp_prob', how = 'inner')
                                    .select('teste')
                                    .collect()[0][0])
            print('best feature to choose --> ',new_test)

            if  self._log_results == True:
                fig = plt.figure()

                #show table
                alfas_betas.show()
                df_pandas = alfas_betas.toPandas()

                for i in range(0, len(df_pandas)):
                    
                    alfa_value = df_pandas.loc[i,'alfa']
                    beta_value = df_pandas.loc[i,'beta']

                    #generate the value
                    x = np.linspace(beta.ppf(0.01,a = alfa_value, b = beta_value),
                                    beta.ppf(0.99,a = alfa_value, b = beta_value),1000)
                    
                    #plot the beta distribution of the experiment
                    col = (np.random.random(), np.random.random(), np.random.random())
                    plt.plot(x, beta.pdf(x, alfa_value, beta_value), 'r-', color = col, label = i)
                    plt.title(f'beta distribution')
                plt.figure(figsize=(8,5))
                plt.show(block = False)
                return new_test
        else:
            #Random selection
            proba_iguais = 1/len(self._variable_test)
            new_test = random.choice(self._variable_test)
            if  self._log_results == True:
                print("Exploration - The best next feature to choose is", new_test)
                print("We dont define priors, so we choose a random option")
        
            return new_test