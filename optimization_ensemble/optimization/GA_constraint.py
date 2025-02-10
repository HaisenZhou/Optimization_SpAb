import numpy as np

from pathlib import Path


def normalize_data(x):
    '''
    Normalize the data, to make sure the sum of each row is 1
    '''
    x = x / x.sum(axis=1, keepdims=True)
    return x

class constraint_genetic_algorithm:
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            num_candidates: int,
            top_k_fraction: float = 0.1,
            top_k_number: int = 8,
            composition_mutation_rate: float = 0.1,
            composition_noise: float = 0.2, # noise added on composition mutation
            gen_switch_mutation_rate: float = 0.1,
            **kwargs
            ):
        
        '''
            input_path: path to the input file
            fitness_path: path to the fitness file
            num_candidates: number of candidates in each generation including top k parents and offspring
            top_k_fraction: fraction of the population that will be selected as parents
            d: number of genes in each chromosome, related to the accuracy of the composition
            mutation_rate: mutation rate
            return best solution
        '''
        
        self.x = x
        self.fitness = y
        self.top_k_number = top_k_number    
        self.num_candidates = num_candidates
        self.top_k_fraction = top_k_fraction
        self.composition_noise = composition_noise
        self.composition_mutation_rate = composition_mutation_rate
        self.gen_switch_mutation_rate = gen_switch_mutation_rate
        # inequality constraints (X[indices[i]] * coefficients[i]) > rhs
        self.constraints = [(np.array([0, 0, 0, 0, -1, 0, 0, 0]), -0.4), (np.array([1, 1, 1, 1, 0, -1, 1, -1]), 0)]
        
        print("Initialization complete")
        
    def selection(
            self,
            fitness: np.ndarray, 
            k: int
            ) -> np.ndarray:
        '''
            choose top k parents

            fitness: 1 x d array
            k: number of parents to be chosen
        '''
        parents = np.argsort(fitness)[-k:]
        print("selection complete")
        return parents
        

    def crossover(
            self,
            parents: np.ndarray,
            offspring_size: int
            ) -> np.ndarray:
        '''
            select 2 parents and randomly select genes from each parent for crossover
            
            parents: parents
            offspring_size: size of offspring
            
        '''
        parents = normalize_data(parents)
        offspring = np.zeros((offspring_size,parents.shape[1]))

        for k in range(offspring_size):
            # randomly select 2 parents
            parent_indices = np.random.choice(parents.shape[0], size=2, replace=False)
            parent1_idx, parent2_idx = parent_indices
            # randomly decide the number of genes from parent1 
            num_genes_parent1 = np.random.randint(1, self.x.shape[1])
            # randomly select genes to pass to offspring based on the frequence
            genes_parent1 = np.random.choice(np.arange(parents.shape[1]), p=parents[parent1_idx],size=num_genes_parent1)
            genes_parent2 = np.random.choice(np.arange(parents.shape[1]), p=parents[parent2_idx],size=self.x.shape[1]-num_genes_parent1)
            # pick the genes from parent1 and parent2 and put them into offspring
            # if there are same gene from parent1 and parent2, average them
            # pick the genes from parent1 and parent2 and put them into offspring
            offspring[k,genes_parent1] = parents[parent1_idx,genes_parent1]
            offspring[k,genes_parent2] = parents[parent2_idx,genes_parent2]
            common_genes = np.intersect1d(genes_parent1, genes_parent2)
            for gene in common_genes:
                offspring[k, gene] = (parents[parent1_idx, gene] + parents[parent2_idx, gene]) / 2
        offspring = normalize_data(offspring.round(2))
        if np.isnan(offspring).any():
            print('crossover')
        print("corssover complete")
        return offspring.round(2)
        
        
    def composition_mutation(
            self, 
            offspring: np.ndarray
            ) -> np.ndarray:
        
        '''
            randomly add gaussian noise to the genes
        '''
        mutated_offspring = offspring
        for idx in range(mutated_offspring.shape[0]):
            # pick the non-zero genes
            non_zero_genes = np.nonzero(mutated_offspring[idx])[0]
            for non_zero_gene in non_zero_genes:
                while True:
                    if np.random.uniform(0,1) < self.composition_mutation_rate:
                        # randomly add a gaussian noise to the gene
                        mutated_offspring[idx,non_zero_gene] = mutated_offspring[idx,non_zero_gene] + np.random.normal(0, self.composition_noise)
                        if mutated_offspring[idx,non_zero_gene] < 0:
                            mutated_offspring[idx,non_zero_gene] = 0.01 
                    mutated_offspring[idx] = mutated_offspring[idx]/mutated_offspring[idx].sum()
                    if self.is_candidate_valid(mutated_offspring[idx]):
                        break
        print("mutation complete")
        return mutated_offspring.round(2)
        
    
    def gen_switch_mutation(
            self,
            offspring: np.ndarray     
            ) -> np.ndarray:
        '''
            randomly switch the genes
        '''
        mutated_offspring = offspring
        for idx in range(mutated_offspring.shape[0]):
            while True:
                if np.random.uniform(0,1) < self.gen_switch_mutation_rate:
                    # randomly select a non-zero gene
                    non_zero_genes = np.nonzero(mutated_offspring[idx])[0]
                    gene_to_switch = np.random.choice(non_zero_genes)
                    # randomly select a gene to switch
                    gene_to_switch_with = np.random.choice(np.arange(offspring.shape[1]))
                    if gene_to_switch == gene_to_switch_with:
                        continue
                    # switch the genes, add the gene_to_switch to gene_to_switch_with together
                    mutated_offspring[idx,gene_to_switch_with] = mutated_offspring[idx,gene_to_switch] + mutated_offspring[idx,gene_to_switch_with]
                    mutated_offspring[idx,gene_to_switch] = 0
                mutated_offspring[idx] = mutated_offspring[idx]/mutated_offspring[idx].sum()
                if self.is_candidate_valid(mutated_offspring[idx]):
                    break
                
        print("gen_switch_mutation complete")
        return mutated_offspring.round(2)
    
    def is_candidate_valid(self, candidate):
        for constraint in self.constraints:
            if np.dot(constraint[0], candidate) < constraint[1]:
                return False
        return True
    
    def random_mutation(self) -> np.ndarray:
        '''
            randomly generate a candidate with all elements are positive and sum to 1
        '''
        rand_candidate = np.random.rand(self.x.shape[1])
        rand_candidate = rand_candidate / rand_candidate.sum()
        # if the rand_candidate is out of the constraints, generate a new one
        while not self.is_candidate_valid(rand_candidate):
            rand_candidate = np.random.rand(self.x.shape[1])
            rand_candidate = rand_candidate / rand_candidate.sum()
        print("random mutation complete")
        return rand_candidate
    
    def propose_new_candidates(self) -> np.ndarray:

        np.random.seed(42)

        # top-k parents
        k = 10
        
        offspring_size = self.num_candidates
        offspring_size_direct_mutation = int(offspring_size * 0.5)
    
        offspring_size_crossover_mutation = offspring_size - offspring_size_direct_mutation
        
        fitness = self.fitness
        population = self.x

        # select top-k parents from parents pool
        selected_parents_idx = self.selection(fitness, k)
        crossover_offspring = self.crossover(population[selected_parents_idx], offspring_size_crossover_mutation)
        mutated_crossover_offspring = self.composition_mutation(crossover_offspring)
        mutated_crossover_offspring = self.gen_switch_mutation(mutated_crossover_offspring)
        
        #randomly select parents from selected_parents_idx offspring_size_direct_mutation times
        direct_mutation_offspring = population[np.random.choice(selected_parents_idx, offspring_size_direct_mutation)]
        # direct composition mutation of the selected parents
        direct_mutation_offspring = self.composition_mutation(direct_mutation_offspring)
        
        # combine the offspring from crossover and direct mutation
        mutated_offspring = np.vstack((mutated_crossover_offspring, direct_mutation_offspring))


        for idx in range(mutated_offspring.shape[0]):

            # combine the previous candidates to the candidates before the current candidate
            previous_data = mutated_offspring[:idx]
            # combine self.x and previous_data
            previous_data = np.vstack((self.x, previous_data)) 
            # Check if the mutated_offspring is the same as the previous data
            while True:
                is_similar_to_any = False  # Initialize flag as False

                for previous in previous_data:
                    if (mutated_offspring[idx] == previous).all():  # Compare the offspring with each previous candidate
                        is_similar_to_any = True  # Set flag to True if similar
                        break  # Break the loop as soon as a similar component is found

                if is_similar_to_any:
                    mutated_offspring[idx] = self.random_mutation()
                    continue
                else:
                    break  # Break the while loop if the offspring is not similar to any previous candidates

        population_next = mutated_offspring.round(2) 
        population_next = normalize_data(population_next)
        print("propose_new_candidates complete")
        return population_next.round(2)


    
    def propose_new_candidates_with_replacement(self):
        valid_candidates = []
        attempts = 0
        max_attempts = 1000  # maximum number of attempts to generate valid candidates
    
        while len(valid_candidates) < self.num_candidates and attempts < max_attempts:
            # generate new candidates
            new_candidates = self.propose_new_candidates()
    
            for candidate in new_candidates:
                if self.is_candidate_valid(candidate):
                    valid_candidates.append(candidate)
                else:
                    # if the candidate is invalid, try to mutate it
                    new_candidate = self.random_mutation()
                    if self.is_candidate_valid(new_candidate):
                        valid_candidates.append(new_candidate)
    
                if len(valid_candidates) >= self.num_candidates:
                    break
    
            attempts += 1
    
        if attempts >= max_attempts:
            raise Exception("Failed to generate enough valid candidates after {} attempts.".format(max_attempts))
    
        return np.array(valid_candidates)
