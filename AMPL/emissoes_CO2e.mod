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


# -----------------
# Variáveis
# -----------------

var Alpha_est >= 0;
var Beta_est;
var Gamma_est;
var Omega_est;
var Mu_est;

var Alpha_din_ant >= 0;
var Beta_din_ant;
var Gamma_din_ant;
var Omega_din_ant;
var Mu_din_ant;

var Alpha_din_pos >= 0;
var Beta_din_pos;
var Gamma_din_pos;
var Omega_din_pos;
var Mu_din_pos;


# Emissões horárias estimadas

var Emissao_horaria_est {a in ANO, h in HORARIO[a]} =
    Alpha_est*pg[a,h]^2
  + Beta_est*pg[a,h]
  + Gamma_est
  + Omega_est*exp(Mu_est*pg[a,h]);


var Emissao_horaria_din {a in ANO, h in HORARIO[a]} =
    Alpha_est*pg[a,h]^2
  + Beta_est*pg[a,h]
  + Gamma_est
  + Omega_est*exp(Mu_est*pg[a,h])

  + Alpha_din_ant*delta_pg_ant[a,h]^2
  + Beta_din_ant*delta_pg_ant[a,h]
  + Gamma_din_ant
  + Omega_din_ant*exp(Mu_din_ant*delta_pg_ant[a,h])

  + Alpha_din_pos*delta_pg_pos[a,h]^2
  + Beta_din_pos*delta_pg_pos[a,h]
  + Gamma_din_pos
  + Omega_din_pos*exp(Mu_din_pos*delta_pg_pos[a,h]);


# -----------------
# Funções objetivo
# -----------------

# MSE: mean squared error

minimize MSE_est:
    1 / (sum {a in ANO} card(HORARIO[a]))
    * sum {a in ANO}
        ( sum {h in HORARIO[a]} Emissao_horaria_est[a,h] - emissoes[a] )^2;


minimize MSE_din:
    1 / (sum {a in ANO} card(HORARIO[a]))
    * sum {a in ANO}
        ( sum {h in HORARIO[a]} Emissao_horaria_din[a,h] - emissoes[a] )^2;


# -----------------
# Problemas
# -----------------

problem Regressao_Estatica:
   MSE_est,
   Emissao_horaria_est,
   Alpha_est,
   Beta_est,
   Gamma_est,
   Omega_est,
   Mu_est;


problem Regressao_Dinamica:
   MSE_din,
   Emissao_horaria_din,
   Alpha_est,
   Beta_est,
   Gamma_est,
   Omega_est,
   Mu_est,
   Alpha_din_ant,
   Beta_din_ant,
   Gamma_din_ant,
   Omega_din_ant,
   Mu_din_ant,
   Alpha_din_pos,
   Beta_din_pos,
   Gamma_din_pos,
   Omega_din_pos,
   Mu_din_pos;
