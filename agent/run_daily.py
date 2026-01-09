import csv
import os
import sqlite3
import sys
import time

# Importa a classe do arquivo agent.py
from agent import AgenteConselheiroDeAcoes

CSV_FILE = "./LSTM/data/finance_data.csv"
DB_FILE = "memory.db"


def print_warning(path: str) -> None:
    """
    Imprime uma mensagem de aviso para o usuário se o arquivo especificado existir.

    Parameters
    ----------
    path : str
        Caminho do arquivo a ser verificado.

    Returns
    -------
    None
    """
    if os.path.exists(path) and path == "memory.db":
        print("--- Verifique o arquivo 'memory.db' para ver o histórico completo do agente  ---")
    if os.path.exists(path) and path == "policy.json":
        print("--- Verifique o arquivo 'policy.json' para ver a política atual do agente ------")
    if os.path.exists(path) and path == "metrics.json":
        print("--- Verifique o arquivo 'metrics.json' para ver o desempenho do agente ---------")


def csv_read(caminho_arquivo: str) -> tuple[list[float], str]:
    """
    Lê o arquivo CSV e retorna uma tupla contendo:
    - uma lista com todo o histórico de preços (floats).
    - o nome da ação (string) extraído do cabeçalho.

    Os dados serão fatiados conforme necessário.
    Ignora a primeira linha (cabeçalho) e a coluna de datas.
    """
    if not os.path.exists(caminho_arquivo):
        print(f"ERRO CRÍTICO: O arquivo '{caminho_arquivo}' não foi encontrado.")
        sys.exit(1)

    precos = []
    nome_da_acao = "Desconhecida"

    try:
        with open(caminho_arquivo, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)

            try:
                header = next(reader)
                if len(header) > 1:
                    nome_da_acao = header[1]
            except StopIteration:
                print("ERRO: Arquivo CSV vazio.")
                sys.exit(1)

            for row in reader:
                if row:  # Evita linhas vazias
                    try:
                        # Pega a coluna 1 (Preço) e converte para float
                        valor = float(row[1])
                        precos.append(valor)
                    except ValueError:
                        continue  # Pula linhas com erro de formatação
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
        sys.exit(1)

    return precos, nome_da_acao


def main() -> None:
    """
    Executa o agente de ações.

    O agente lê o arquivo de dados de mercado, avalia o histórico e decide se
    deve comprar, vender ou manter sua posição atual.

    Se o agente decide comprar ou vender, ele ajusta seu nível de exigência
    (threshold) com base na variação real do mercado. Caso contrário, mantém o
    nível de exigência atual.

    O agente aprende com o que aconteceu de ontem pra hoje e ajusta seu
    nível de exigência com base na variação real.

    O agente decide para amanhã com novo limiar já ajustado.

    O agente executa a decisão com base no histórico completo. O agente, usando
    o modelo LSTM disponível, vai decidir quantos dias usar (ex: os últimos 5
    para volatilidade, os últimos 60 para LSTM).
    """
    print("\n--- EXECUTANDO AGENTE CONSELHEIRO DE AÇÕES ---\n")

    print(f"Lendo arquivo '{CSV_FILE}'...")
    market_history, nome_da_acao = csv_read(CSV_FILE)
    print(f"Ativo Identificado: '{nome_da_acao}'")
    time.sleep(0)

    if len(market_history) < 5:
        print("ERRO: Histórico insuficiente (mínimo 5 dias).")
        return

    preco_ontem = market_history[-2]
    # Recebimento do novo preço de fechamento do mercado HOJE
    preco_hoje = market_history[-1]

    print(f"\n{'=' * 50}")
    print(f">>> Preço de fechamento do dia anterior: ${preco_ontem:.2f}")
    # print(f"{'=' * 50}")
    print(f">>> Preço de fechamento do mercado HOJE: ${preco_hoje:.2f}")
    print(f"{'=' * 50}")
    print()
    time.sleep(0)

    # Instancia o agente
    agente = AgenteConselheiroDeAcoes()

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT action, predicted_price FROM history WHERE outcome_evaluated = 0 ORDER BY id ASC LIMIT 1"
    )
    last_trade = cursor.fetchone()
    conn.close()

    if last_trade:
        last_action, last_pred = last_trade
        print(f"   - No dia anterior, o Agente recomendou: {last_action}")
        print(f"   - Ele esperava que o preço fosse para ${last_pred:.2f}")
    else:
        print("   - Ainda não houve um trade.")

    time.sleep(0)
    # Agente aprende com o que aconteceu de ontem pra hoje
    print("\n[APRENDIZADO DO AGENTE COM BASE NO PASSADO]\n")
    time.sleep(0)

    # O agente verifica se tinha alguma recomendação pendente e usa o preço de hoje
    # para saber se acertou ou errou.
    msg_aprendizado, var_real = agente.learn(preco_hoje)

    if msg_aprendizado:
        # Se houve ajuste, mostramos o feedback
        sinal_real = "SUBIU" if var_real > 0 else "CAIU"
        print(f"   - O mercado {sinal_real} {var_real * 100:.2f}% desde ontem.")
        print(f"   - Ajuste do agente: {msg_aprendizado}")
        if msg_aprendizado == "Mantendo rigor. Nenhuma oportunidade encontrada.":
            print(
                f"   - Nível de Exigência mantido (Threshold): {agente.policy['threshold'] * 100:.4f}%"
            )
        else:
            print(
                f"   - Novo Nível de Exigência (Threshold): {agente.policy['threshold'] * 100:.4f}%"
            )
    else:
        # Se não houve trade ou decisão pendente
        print("   - Nenhuma operação pendente de avaliação.")

    time.sleep(0)
    print()
    print("-" * 50)
    print()
    # Agente decide para amanhã com novo limiar já ajustado
    # Passamos o histórico completo. O agente, usando o modelo LSTM disponível,
    # vai decidir quantos dias usar (ex: os últimos 5 para volatilidade, os últimos 60 para LSTM).
    print("[PRÓXIMA ANÁLISE DO AGENTE]\n")
    time.sleep(0)

    # Executa a decisão
    acao, preco_previsto, delta_previsto = agente.decide(market_history)

    # Recalculamos a volatilidade aqui só para exibir ao usuário o que o agente viu internamente
    volatilidade_atual = agente._calcular_volatilidade(market_history)

    # Exibe os dados que o "cérebro" do agente processou
    print(f"   - Volatilidade Recente (Risco): {volatilidade_atual * 100:.2f}%")
    print(f"   - LSTM Prevê para Amanhã:      ${preco_previsto:.2f}")
    print(f"   - Variação Esperada (Delta):   {delta_previsto * 100:.2f}%")

    print("\nGerando recomendação...")
    time.sleep(0)

    # Conclusão final
    indicador = acao

    print()
    print("-" * 50)
    print()
    print(f"   => DECISÃO FINAL DO AGENTE PARA HOJE: {indicador}")

    threshold_atual = agente.policy["threshold"]
    # Como o agente usa um "Threshold Efetivo" (com volatilidade) dentro do decide,
    # vamos estimar se foi a volatilidade ou a política que segurou o trade
    motivo = "Aguardando oportunidade clara."

    if acao != "MANTER":
        motivo = f"Variação de {delta_previsto * 100:.2f}% supera o risco."
    elif abs(delta_previsto) > threshold_atual:
        motivo = "Sinal existe, mas volatilidade alta forçou cautela."
    else:
        motivo = "Variação esperada irrelevante frente ao risco atual."

    print(f"      (Motivo: {motivo})")

    print("\n--- FIM DA EXECUÇÃO DO AGENTE CONSELHEIRO DE AÇÕES ---\n")

    print("-" * 80)
    print_warning("memory.db")
    print_warning("policy.json")
    print_warning("metrics.json")
    print("-" * 80)


if __name__ == "__main__":
    main()
