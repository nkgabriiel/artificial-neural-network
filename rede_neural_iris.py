"""
Rede Neural Artificial (MLP) com uma camada escondida, implementada do zero
com NumPy, para classificar as espécies de flores do conjunto de dados Iris.

Arquitetura: 4 (entrada) -> 8 (ReLU) -> 3 (Softmax)
Treinamento: Gradiente Descendente (batch completo) com Cross-Entropy Loss.

O Scikit-Learn é usado apenas para carregar, dividir e padronizar os dados.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Hiperparâmetros
# ---------------------------------------------------------------------------
SEMENTE = 42
N_ENTRADA = 4
N_ESCONDIDA = 8
N_SAIDA = 3
TAXA_APRENDIZADO = 0.1
EPOCAS = 1000

np.random.seed(SEMENTE)


# ---------------------------------------------------------------------------
# 1. Pré-processamento dos dados
# ---------------------------------------------------------------------------
def one_hot(y, n_classes):
    """Converte rótulos inteiros (0, 1, 2) em vetores binários de tamanho n_classes."""
    return np.eye(n_classes)[y]


iris = load_iris()
X, y = iris.data, iris.target  # X: (150, 4), y: (150,)

# Divisão 80% treino / 20% teste (estratificada para manter a proporção das classes)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=SEMENTE, stratify=y
)

# Padronização (média 0, desvio padrão 1). O scaler é ajustado apenas no treino
# e reaplicado no teste, para que nenhuma informação do teste vaze no treinamento.
scaler = StandardScaler()
X_treino = scaler.fit_transform(X_treino)
X_teste = scaler.transform(X_teste)

# One-Hot Encoding dos rótulos
Y_treino = one_hot(y_treino, N_SAIDA)  # (120, 3)
Y_teste = one_hot(y_teste, N_SAIDA)    # (30, 3)


# ---------------------------------------------------------------------------
# 2. Funções de ativação e de custo
# ---------------------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)


def relu_derivada(z):
    return (z > 0).astype(float)


def softmax(z):
    # Subtrai o máximo de cada linha para estabilidade numérica (evita overflow em exp)
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(Y_real, Y_prev):
    """Entropia cruzada média: L = -(1/m) * soma(y * log(y_hat))."""
    m = Y_real.shape[0]
    Y_prev = np.clip(Y_prev, 1e-12, 1.0)  # evita log(0)
    return -np.sum(Y_real * np.log(Y_prev)) / m


# ---------------------------------------------------------------------------
# Inicialização aleatória dos pesos e vieses
# (inicialização de He, adequada para ReLU: N(0, 1) * sqrt(2 / n_entrada))
# ---------------------------------------------------------------------------
W1 = np.random.randn(N_ENTRADA, N_ESCONDIDA) * np.sqrt(2.0 / N_ENTRADA)    # (4, 8)
b1 = np.zeros((1, N_ESCONDIDA))                                            # (1, 8)
W2 = np.random.randn(N_ESCONDIDA, N_SAIDA) * np.sqrt(2.0 / N_ESCONDIDA)    # (8, 3)
b2 = np.zeros((1, N_SAIDA))                                                # (1, 3)


def forward(X):
    """Forward pass: retorna as pré-ativações e ativações de cada camada."""
    Z1 = X @ W1 + b1      # (m, 8)
    A1 = relu(Z1)         # (m, 8)
    Z2 = A1 @ W2 + b2     # (m, 3)
    A2 = softmax(Z2)      # (m, 3) -> probabilidades de cada classe
    return Z1, A1, Z2, A2


def acuracia(Y_real, Y_prev):
    return np.mean(np.argmax(Y_real, axis=1) == np.argmax(Y_prev, axis=1))


# ---------------------------------------------------------------------------
# 3. Treinamento via Gradiente Descendente
# ---------------------------------------------------------------------------
m = X_treino.shape[0]

print(f"Treinando por {EPOCAS} épocas (taxa de aprendizado = {TAXA_APRENDIZADO})\n")
for epoca in range(1, EPOCAS + 1):
    # Forward pass
    Z1, A1, Z2, A2 = forward(X_treino)

    # Cálculo da perda
    perda = cross_entropy(Y_treino, A2)

    # Backward pass (regra da cadeia)
    # Para Softmax + Cross-Entropy, dL/dZ2 simplifica para (A2 - Y) / m
    dZ2 = (A2 - Y_treino) / m                      # (m, 3)
    dW2 = A1.T @ dZ2                               # (8, 3)
    db2 = np.sum(dZ2, axis=0, keepdims=True)       # (1, 3)

    dA1 = dZ2 @ W2.T                               # (m, 8)
    dZ1 = dA1 * relu_derivada(Z1)                  # (m, 8)
    dW1 = X_treino.T @ dZ1                         # (4, 8)
    db1 = np.sum(dZ1, axis=0, keepdims=True)       # (1, 8)

    # Atualização dos parâmetros: theta = theta - lr * gradiente
    W2 -= TAXA_APRENDIZADO * dW2
    b2 -= TAXA_APRENDIZADO * db2
    W1 -= TAXA_APRENDIZADO * dW1
    b1 -= TAXA_APRENDIZADO * db1

    if epoca == 1 or epoca % 100 == 0:
        print(f"Época {epoca:4d} | Perda: {perda:.4f} | "
              f"Acurácia (treino): {acuracia(Y_treino, A2) * 100:6.2f}%")


# ---------------------------------------------------------------------------
# 4. Avaliação do modelo no conjunto de teste
# ---------------------------------------------------------------------------
_, _, _, A2_teste = forward(X_teste)
perda_teste = cross_entropy(Y_teste, A2_teste)
acc_teste = acuracia(Y_teste, A2_teste)

print("\n" + "=" * 50)
print(f"Perda no conjunto de teste:    {perda_teste:.4f}")
print(f"Acurácia no conjunto de teste: {acc_teste * 100:.2f}%")
print("=" * 50)
