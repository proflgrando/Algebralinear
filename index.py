import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# Configuração da página
st.set_page_config(
    page_title="Aula: Álgebra Linear e Regressão",
    layout="wide"
)

# --- Definições Globais de Dados ---
# Matriz A e vetor b para o sistema linear
MATRIZ_A = np.array([
    [10, 7, 8, 7],
    [7, 5, 6, 5],
    [8, 6, 10, 9],
    [7, 5, 9, 10]
])
VETOR_B = np.array([32, 23, 33, 31])

# Dados para a Regressão
X_VETOR = np.array([1, 2, 3, 4, 5])
Y_VETOR = np.array([2.1, 3.9, 6.1, 7.8, 10.2]) # Quase y = 2x + 0 (mas com ruído)


# --- Funções das Páginas ---

def pagina_introducao():
    st.title("Aula Interativa: Álgebra Linear e Regressão 📈")
    st.markdown("""
    Bem-vindo! Este aplicativo demonstra como usar **NumPy** para tarefas de Álgebra Linear e como ele se compara ao **Scikit-learn** para Regressão Linear.
    
    Use o menu ao lado para navegar pelas seções da aula.
    """)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/1280px-NumPy_logo_2020.svg.png",
        width=200
    )


def pagina_fundamentos_matrizes():
    st.header("PARTE 1: Fundamentos de Matrizes com NumPy")
    st.markdown("O NumPy é a nossa 'calculadora' para operações com matrizes (arrays 2D).")

    st.subheader("1.1: Criação de Matrizes")
    st.write("Vamos definir nossa matriz `A` (do sistema linear) e uma matriz Identidade `I`:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Matriz A (4x4):")
        st.code(f"A = {MATRIZ_A}")
    with col2:
        st.write("Matriz Identidade I (4x4):")
        B_identidade = np.identity(4)
        st.code(f"I = {B_identidade}")

    st.subheader("1.2: Operações Básicas com Matrizes")

    st.markdown("Veja o que acontece quando realizamos operações:")
    
    with st.expander("Soma (A + I)"):
        soma = MATRIZ_A + B_identidade
        st.code(soma)

    with st.expander("Transposição (A.T)"):
        A_transposta = MATRIZ_A.T
        st.code(A_transposta)
        
    with st.expander("Produto Matricial (A @ I)"):
        st.markdown("Usamos `@` para produto matricial. (Note que `A @ I` é igual a `A`).")
        produto = MATRIZ_A @ B_identidade
        st.code(produto)

    with st.expander("Inversa da Matriz (A⁻¹)"):
        st.markdown("""
        A inversa `A⁻¹` é a matriz tal que `A @ A⁻¹ = I`.
        Usamos `np.linalg.inv(A)`.
        """)
        try:
            A_inversa = np.linalg.inv(MATRIZ_A)
            st.write("Matriz A Inversa (A⁻¹):")
            st.code(A_inversa)
            
            st.write("Verificação (A @ A⁻¹) - Deve ser a Matriz Identidade:")
            verificacao = MATRIZ_A @ A_inversa
            st.code(np.round(verificacao)) # Arredondamos para limpar erros de precisão

        except np.linalg.LinAlgError:
            st.error("A matriz A não possui inversa (é singular).")


def pagina_sistemas_lineares():
    st.header("PARTE 2: Resolvendo Sistemas Lineares (Ax = b)")
    st.markdown("Queremos resolver o sistema 4x4:")
    
    # Exibe a equação em formato LaTeX
    st.latex(r"""
    \begin{cases}
    10x + 7y + 8z + 7w = 32 \\
    7x + 5y + 6z + 5w = 23 \\
    8x + 6y + 10z + 9w = 33 \\
    7x + 5y + 9z + 10w = 31
    \end{cases}
    """)
    
    st.subheader("Definindo A e b")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Matriz A (Coeficientes):")
        st.code(MATRIZ_A)
    with col2:
        st.write("Vetor b (Resultados):")
        st.code(VETOR_B)
        
    st.subheader("Abordagem 1: Usando a Inversa (Ineficiente)")
    st.markdown("Matematicamente, `x = A⁻¹ @ b`. Vamos calcular:")
    try:
        A_inv = np.linalg.inv(MATRIZ_A)
        x_solucao_1 = A_inv @ VETOR_B
        st.code(f"x = {x_solucao_1}")
    except np.linalg.LinAlgError:
        st.error("Método 1 falhou (matriz singular).")

    st.subheader("Abordagem 2: O Jeito Correto (np.linalg.solve)")
    st.markdown("""
    Usar a inversa é lento e instável. A forma correta e otimizada no NumPy
    é usar `np.linalg.solve(A, b)`.
    """)
    try:
        x_solucao_2 = np.linalg.solve(MATRIZ_A, VETOR_B)
        st.success(f"Solução (x, y, z, w): {x_solucao_2}")
        st.markdown("A resposta é **[1, 1, 1, 1]**! Esta é a forma preferida.")
    except np.linalg.LinAlgError:
        st.error("Método 2 (solve) falhou (matriz singular).")


def pagina_regressao_numpy():
    st.header("PARTE 3: Regressão com NumPy (A Equação Normal)")
    st.markdown("Queremos encontrar a linha `y = B0 + B1*x` que melhor se ajusta aos nossos dados.")
    
    st.subheader("A Fórmula Mágica (Equação Normal)")
    st.latex(r"\theta = (X^T X)^{-1} X^T y")
    st.markdown(r"Onde $\theta$ (theta) é o vetor de parâmetros `[B0, B1]`.")

    st.subheader("1. Nossos Dados")
    col1, col2 = st.columns(2)
    with col1:
        st.write("X (Tamanho m²):")
        st.code(X_VETOR)
    with col2:
        st.write("y (Preço):")
        st.code(Y_VETOR)

    st.subheader("2. Preparar a Matriz Design (X_design)")
    st.markdown("""
    A fórmula precisa de uma coluna de '1s' em X para calcular o intercepto (B0).
    Usamos `np.c_[np.ones(...), X_vetor]` para criar:
    """)
    X_design = np.c_[np.ones(X_VETOR.shape[0]), X_VETOR]
    st.code(X_design)
    
    st.subheader("3. Aplicando a Fórmula Passo a Passo")
    
    with st.expander("Passo 1: X Transposto (XT)"):
        XT = X_design.T
        st.code(XT)
        
    with st.expander("Passo 2: XT @ X"):
        XTX = XT @ X_design
        st.code(XTX)
        
    with st.expander("Passo 3: Inversa de (XT @ X)"):
        XTX_inv = np.linalg.inv(XTX)
        st.code(XTX_inv)
        
    with st.expander("Passo 4: XT @ y"):
        XTy = XT @ Y_VETOR
        st.code(XTy)
        
    st.subheader("4. Resultado Final (Theta)")
    st.markdown("`theta = (Passo 3) @ (Passo 4)`")
    theta = XTX_inv @ XTy
    
    st.success(f"Theta (B0, B1): {theta}")
    st.markdown(f"**Equação da Linha: y = {theta[0]:.4f} + {theta[1]:.4f} * x**")


def pagina_regressao_sklearn():
    st.header("PARTE 4: Regressão com Scikit-learn (O Jeito Fácil)")
    st.markdown("""
    O Scikit-learn (sklearn) é a biblioteca padrão para Machine Learning.
    Ela esconde toda a complexidade da Equação Normal em duas linhas: `.fit()` e `.predict()`.
    """)
    
    st.subheader("1. Preparar os Dados para o Sklearn")
    st.markdown("""
    O Sklearn exige que X seja uma matriz 2D (mesmo com uma só feature).
    Usamos `.reshape(-1, 1)` para transformar `[1, 2, 3]` em `[[1], [2], [3]]`.
    """)
    X_sklearn = X_VETOR.reshape(-1, 1)
    st.code(X_sklearn)
    
    st.subheader("2. Criar e Treinar o Modelo")
    st.markdown("Este é o 'coração' do sklearn:")
    st.code("""
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
modelo.fit(X_sklearn, Y_VETOR)
    """)
    # Treinamento real (nos bastidores)
    modelo = LinearRegression()
    modelo.fit(X_sklearn, Y_VETOR)
    
    st.subheader("3. Ver os Resultados")
    intercepto_sklearn = modelo.intercept_
    coeficiente_sklearn = modelo.coef_
    
    st.success(f"Intercepto (B0): {intercepto_sklearn:.4f}")
    st.success(f"Coeficiente (B1): {coeficiente_sklearn[0]:.4f}")
    
    
def pagina_conclusao():
    st.header("Conclusão: NumPy vs. Scikit-learn")
    st.markdown("Vamos comparar os resultados da Regressão Linear:")

    # Recalcula o NumPy
    X_design = np.c_[np.ones(X_VETOR.shape[0]), X_VETOR]
    XT = X_design.T
    XTX = XT @ X_design
    XTX_inv = np.linalg.inv(XTX)
    XTy = XT @ Y_VETOR
    theta = XTX_inv @ XTy
    
    # Recalcula o Sklearn
    X_sklearn = X_VETOR.reshape(-1, 1)
    modelo = LinearRegression()
    modelo.fit(X_sklearn, Y_VETOR)
    intercepto_sklearn = modelo.intercept_
    coeficiente_sklearn = modelo.coef_
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("NumPy (Manual)")
        st.code(f"B0 (Intercepto) = {theta[0]:.4f}\nB1 (Coeficiente) = {theta[1]:.4f}")
        
    with col2:
        st.subheader("Scikit-learn (Automático)")
        st.code(f"B0 (Intercepto) = {intercepto_sklearn:.4f}\nB1 (Coeficiente) = {coeficiente_sklearn[0]:.4f}")

    st.info("Os resultados são idênticos!")
    st.markdown("""
    - O **NumPy** nos dá o **COMO** (os blocos de construção e a matemática).
    - O **Scikit-learn** nos dá o **O QUÊ** (a ferramenta pronta para uso).
    
    Entender o NumPy nos permite entender o que o Scikit-learn faz por baixo dos panos.
    """)

# --- Menu Principal (Sidebar) ---

st.sidebar.title("Menu da Aula")
paginas = {
    "Introdução": pagina_introducao,
    "1. Fundamentos de Matrizes": pagina_fundamentos_matrizes,
    "2. Sistemas Lineares": pagina_sistemas_lineares,
    "3. Regressão com NumPy": pagina_regressao_numpy,
    "4. Regressão com Scikit-learn": pagina_regressao_sklearn,
    "5. Conclusão": pagina_conclusao,
}

selecao = st.sidebar.radio("Selecione a Seção:", list(paginas.keys()))

# Executa a função da página selecionada
pagina_selecionada = paginas[selecao]
pagina_selecionada()
