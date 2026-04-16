# Convenção AMPL:

# Conjuntos (set) em MAIÚSCULAS
# Parâmetros (param) em minúsculas

# -----------------
# Conjuntos
# -----------------

set ANO;
param H {ANO};
set HORARIO {a in ANO} ordered := 0..H[a];

# -----------------
# Parâmetros
# -----------------

param emissoes {ANO}; # em kg de CO2e

param pg {a in ANO, h in HORARIO[a]};
param delta_pg_ant {a in ANO, h in HORARIO[a]};
param delta_pg_pos {a in ANO, h in HORARIO[a]};

param S_base := 100;

# Parâmetro de regularização (lambda)
param lambda := 30;  # ajuste conforme necessidade

# -----------------
# Variáveis
# -----------------

var Alpha >= 0;      # α - Representa emissões que crescem com o quadrado da geração (carga muito alta)
var Beta;            # β - Representa eficiência em carga parcial (pode ser negativo em algumas faixas)
var Gamma;           # γ - Termo constante (pode ser zero ou positivo)
var Omega;           # ω - Amplitude do termo exponencial (emissões não-lineares)
var Mu;              # μ - Taxa de crescimento das emissões em altas cargas

var Alpha_L2 >= 0;
var Beta_L2;
var Gamma_L2;
var Omega_L2;
var Mu_L2;

# -----------------
# Restrições Físicas
# -----------------

# Emissões nunca podem ser negativas
subject to Emissao_nao_negativa {a in ANO, h in HORARIO[a]}:
    Alpha * pg[a,h]^2
    + Beta * pg[a,h]
    + Gamma
    + Omega * exp(Mu * pg[a,h]) >= 0;

subject to Emissao_nao_negativa_L2 {a in ANO, h in HORARIO[a]}:
    Alpha_L2 * pg[a,h]^2
    + Beta_L2 * pg[a,h]
    + Gamma_L2
    + Omega_L2 * exp(Mu_L2 * pg[a,h]) >= 0;


# Emissão zero quando geração é zero (ponto de referência)
subject to Emissao_no_zero:
    Gamma + Omega >= 0;

subject to Emissao_no_zero_L2:
    Gamma_L2 + Omega_L2 >= 0;


# Emissões horárias estimadas
var Emissao_horaria {a in ANO, h in HORARIO[a]} =
    Alpha*pg[a,h]^2
  + Beta*pg[a,h]
  + Gamma
  + Omega*exp(Mu*pg[a,h]);

var Emissao_horaria_L2 {a in ANO, h in HORARIO[a]} =
    Alpha_L2*pg[a,h]^2
  + Beta_L2*pg[a,h]
  + Gamma_L2
  + Omega_L2*exp(Mu_L2*pg[a,h]);


# -----------------
# Funções objetivo
# -----------------

# MSE: mean squared error
minimize MSE:
    1 / (sum {a in ANO} card(HORARIO[a]))
    * sum {a in ANO}
        ( sum {h in HORARIO[a]} Emissao_horaria[a,h] - emissoes[a] )^2;

minimize MSE_L2:
    1 / (sum {a in ANO} card(HORARIO[a]))
    * sum {a in ANO}
        ( sum {h in HORARIO[a]} Emissao_horaria_L2[a,h] - emissoes[a] )^2
    + lambda * (Alpha_L2^2 + Beta_L2^2 + Gamma_L2^2 + Omega_L2^2 + Mu_L2^2);

# -----------------
# Problemas
# -----------------

problem Regressao_Estatica:
   MSE,
   Emissao_horaria,
   Alpha,
   Beta,
   Gamma,
   Omega,
   Mu;

problem Regressao_Estatica_L2:
   MSE_L2,
   Emissao_horaria_L2,
   Alpha_L2,
   Beta_L2,
   Gamma_L2,
   Omega_L2,
   Mu_L2;