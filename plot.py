import matplotlib.pyplot as plt
import numpy as np

def plot_score_optimization(total_dataset_labels=14):
    # Simulação baseada em comportamentos reais de MAE meteorológico
    labels_count = np.arange(1, 11) # Testando de 1 a 10 labels
    

    maes_simulados = [0.85, 0.90, 1.05, 1.15, 1.80, 2.50, 3.20, 4.10, 5.50, 7.00]
    
    scores = []
    for n, mae in zip(labels_count, maes_simulados):
        score = (2.5 / (1 + mae)) * (n / total_dataset_labels) * 100
        scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(labels_count, scores, marker='o', linestyle='-', color='b', linewidth=2)
    
    plt.title('Otimização do Score Level 5: Precisão vs. Quantidade')
    plt.xlabel('Número de Labels Preditas')
    plt.ylabel('Score Final Estimado')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    
    plt.show()

plot_score_optimization()