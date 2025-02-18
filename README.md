# Dissertation-Analyses
Code used to read and analyse the results from the simulations done during my Dissertation.
Latest version: 3plot

- OS CALCULOS SÃO TODOS PARA O CASO 2D (z=0 e vz=0)

- O PLANETA DEFORMADO É O PLANETA b 

- PRECISA DOS FICHEIROS .car .rot e *ParamInit.out*
de uma lista dos valores de tau, de uma lista dos ângulos,
de uma lista da pasta dos dados, de uma lista com C22r

- É NECESSÁRIO FORNECER O VALOR DE C22r QUE SE PRETENDE VISUALIZAR NOS ARGUMENTOS
(por default utiliza C22r = 1E-7)

Para correr o programa:
    python3 allpyplot.py (+)*valores de tau* *dados* *c22r* *c/ ou s/ plots interativos* *c/ ou s/ imgs guardadas"
    
    Ex.: python3 allpyplot.py +1E-1 e_HIGH 0 p s
    
        ->  +: faz prints de debug (sem debug não colocar +)
        ->  1E-1: valor de tau utilizado (separar vários valores por virgulas)
        -> e_HIGH: lê os dados da pasta e_HIGH
        -> 0: todos os valores de c22r
        ->  p: c/ gráficos interativos, mas não os guarda (sem plots colocar 0, para os guardar na pasta)
        -> s: guarda os gráficos na pasta PLOTS

Para correr gravando apenas os graficos numa pasta:
    python3 allpyplot.py 0 0 0 s
