import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv("train 2.csv")

#porcentagem 
def to_percentagem(series):
    return series *100
# Criar gráficos
plt.figure(figsize=(15, 5))

# Gráfico de sobrevivência por gênero
plt.subplot(1, 4, 1)
gender_counts = df.groupby("Sex")["Survived"].mean()
plt.pie(to_percentagem(gender_counts), labels=gender_counts.index, autopct="%.1f%%", colors=["lightblue", "blue"], startangle=90)
plt.title("Taxa de Sobrevivência por Gênero")


# Gráfico de sobrevivência por classe social
plt.subplot(1, 4, 2)
sns.barplot(x=df['Pclass'], y=df['Survived'], ci=None, palette="Blues")
plt.title("Taxa de Sobrevivência por Classe Social")
plt.xlabel("Classe Social")
plt.ylabel("Taxa de Sobrevivência (%)")

# Criar faixas etárias
bins = [0, 12, 18, 30, 50, 100]
labels = ["Criança", "Adolescente", "Jovem Adulto", "Adulto", "Idoso"]
df["Faixa Etária"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

# Gráfico de sobrevivência por faixa etária
plt.subplot(1, 4, 3)
sns.barplot(x=df['Faixa Etária'], y=df['Survived'], ci=None, palette="Blues")
plt.title("Taxa de Sobrevivência por Faixa Etária")
plt.xlabel("Faixa Etária")
plt.ylabel("Taxa de Sobrevivência (%)")
plt.xticks(rotation=45)

plt.subplot(1, 4, 4)
survival_counts = df["Survived"].value_counts(normalize=True) * 100
plt.pie(survival_counts, labels=["Não Sobreviveu", "Sobreviveu"], autopct="%.1f%%", colors=["lightblue", "blue"], startangle=90)
plt.title("Taxa Geral de Sobrevivência")

# Exibir os gráficos
plt.tight_layout()
plt.show()