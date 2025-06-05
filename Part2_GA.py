from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
from sklearn.model_selection import train_test_split

# === 1. Φόρτωση δεδομένων ===
df = pd.read_csv('/content/alzheimers_disease_data.csv')
X = df.drop(columns=['Diagnosis', 'PatientID', 'DoctorInCharge'])
y = df['Diagnosis']

# === 2. Φόρτωση scaler από το Μέρος Α ===
scaler = joblib.load('/content/drive/MyDrive/ColabNotebooks/scaler.pkl')
X_scaled = scaler.transform(X)

# === 3. Φόρτωση μοντέλου από το Μέρος Α ===
model = load_model('/content/drive/MyDrive/ColabNotebooks/best_model.keras')

# === 4. Διαχωρισμός σε train/test ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) #Χρησιμοποιώ μόνο τα test δεδομένα για την αξιολόγηση του κάθε ατόμου

input_dim = X_train.shape[1]  # 34 χαρακτηριστικά

# === 5. Συνάρτηση αξιολόγησης ατόμου ===
def evaluate_individual(individual, model):
    masked_X_test = X_test * individual
    loss, accuracy = model.evaluate(masked_X_test, y_test, verbose=0)
    penalty = 0.01 * np.sum(individual)
    fitness = accuracy - penalty
    return fitness

# === 6. ΓΑ Ρυθμίσεις ===
POP_SIZE = 200
CROSSOVER_RATE = 0.1
MUT_RATE = 0.01
N_FEATURES = input_dim

max_generations = 100
patience = 6
min_improvement = 0.005

all_runs = []
generations_per_run = []

for run in range(3):
    population = np.random.randint(0, 2, (POP_SIZE, N_FEATURES))
    best_fitness = -np.inf
    generations_without_improvement = 0
    best_fitness_per_generation = []

    # === 7. ΓΑ: Κύριος βρόχος ===
    for gen in range(max_generations):
        fitness_scores = [evaluate_individual(ind, model) for ind in population]
        current_best = max(fitness_scores)
        improvement = current_best - best_fitness

        current_best_idx = np.argmax(fitness_scores)
        current_best = fitness_scores[current_best_idx]
        best_individual = population[current_best_idx].copy()

        if improvement < min_improvement:
            generations_without_improvement += 1
        else:
            generations_without_improvement = 0
            best_fitness = current_best

        if generations_without_improvement >= patience:
            print(f"Stopped at generation {gen + 1} due to early convergence")
            generations_per_run.append(gen + 1)
            break

        if generations_without_improvement < patience:
            generations_per_run.append(max_generations)

        # Tournament selection
        selected = []
        for _ in range(POP_SIZE):
            i1, i2 = np.random.choice(range(POP_SIZE), 2)
            winner = population[i1] if fitness_scores[i1] > fitness_scores[i2] else population[i2]
            selected.append(winner)

        # Crossover & Mutation
        new_population = []
        for i in range(0, POP_SIZE, 2):
            p1 = selected[i]
            p2 = selected[i + 1]

            if np.random.rand() < CROSSOVER_RATE:
                mask = np.random.randint(0, 2, size=N_FEATURES)
                child1 = np.where(mask == 1, p1, p2)
                child2 = np.where(mask == 1, p2, p1)
            else:
                child1 = np.copy(p1)
                child2 = np.copy(p2)

            # Mutation
            for child in [child1, child2]:
                for j in range(N_FEATURES):
                    if np.random.rand() < MUT_RATE:
                        child[j] = 1 - child[j]

                new_population.append(child)

        # === Ελιτισμός ===
        new_population = np.array(new_population)
        new_fitness_scores = [evaluate_individual(ind, model) for ind in new_population]
        worst_idx = np.argmin(new_fitness_scores)
        new_population[worst_idx] = best_individual  # Replace worst with best from previous gen

        population = new_population
        best_ind = population[np.argmax(fitness_scores)]
        print(f"Generation {gen + 1}: Best Fitness = {best_fitness:.4f} | Features used: {np.sum(best_ind)}")

        best_fitness_per_generation.append(best_fitness)

    all_runs.append(best_fitness_per_generation)

    # === 8. Τελική αξιολόγηση ===
    final_mask = best_ind
    masked_X_test = X_test * final_mask
    loss, accuracy = model.evaluate(masked_X_test, y_test, verbose=0)
    print(f"\nFinal accuracy with selected features: {accuracy:.4f}")
    print("\nΚαλύτερο άτομο:")
    print(f"Features used: {np.where(best_ind == 1)[0]}")

best_final_fitnesses = [max(run) for run in all_runs]
mean_best_fitness_overall = np.mean(best_final_fitnesses)
mean_generations = np.mean(generations_per_run)
print(f"🧬 Μέσος αριθμός γενεών μέχρι τη σύγκλιση: {mean_generations:.2f}")
print(f"\n📊 Μέσος όρος του καλύτερου fitness από κάθε επανάληψη: {mean_best_fitness_overall:.4f}")

# === 9. Padding και Μέσος όρος ===
max_len = max(len(run) for run in all_runs)
for i in range(len(all_runs)):
    all_runs[i] += [all_runs[i][-1]] * (max_len - len(all_runs[i]))

mean_fitness = np.mean(all_runs, axis=0)

# === 10. Plot ===
plt.plot(mean_fitness)
plt.xlabel('Generation')
plt.ylabel('Mean Best Fitness')
plt.title('Evolution of Mean Best Fitness Over Generations')
plt.grid(True)
plt.show()
