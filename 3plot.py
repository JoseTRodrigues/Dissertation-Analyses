# -*- coding: utf-8 -*-
"""
VERSÃO: 25/7/2024
    Alterações:
        Major
        - Correção das unidades do R_p (de R_sun para AU)
        - Correção do cálculo do semi-eixo e da excentricidade (considerando a precessão)
        - Cálculo da Energia disspada por unidade de área (E/4piR^2) em unidades SI
        - Cálculo de n2/n1 (e substituição pelo semi-eixo nos gráficos)
        - Incluidas excentricidades dos 2 planetas no gráfico
        - Retirado gráfico de n para n = 1 e substituido pelo outro valor de C22r 

        minor
        - Retirada linha de resonancia de rotação 1.5
        - 


PLOTS EM PYTHON

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
    
@author: José Rodrigues
"""

# %% IMPORTS

# argv = ['','0.01','e_HIGH','p']
from sys import argv

print('\nargv: ', argv,'\n')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


#---------------- FUNÇÕES -------------------------------------------------------
def flush_print (*args):
    if argv[1][0] == '+':
        import datetime
        now = datetime.datetime.now()
        timestamp = now.strftime('%H:%M:%S.%f')
        print (timestamp,  *args, flush=True)
        return

def movingaverage(data, window_width):
    """
    Computes moving average of an array
    """
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

# def derivative(df,dx):
#     """
#     Computes the derivative 
#     using a five-point method

#     df: Pandas Dataframe

#     dx: Step between data

#     """
#     result = np.array([])
#     for i in range(2,len(df)-2):
#         result = np.append(result, 
#             (df[i-2]-8*df[i-1]+8*df[i+1]-df[i+2])/(12*dx)
#             )
    
#     return result
#-------------------------------------------------------------------------------------------------------

flush_print('Initiating...\n')


#%% INPUTS
"================== ASPECTO DOS GRÁFICOS =============="
# ASPECTO DAS LETRAS
plt.rc('font', size = 12)
plt.rc('axes', titleweight = 'bold')

#TAMANHO DAS IMAGENS
# altura  = 1080
# largura = 1920

altura = 718
largura = 1420

altura  = altura - 66
largura = largura - 70
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
figsize_init = (largura*px,altura*px)

if 's' in argv:
    altura  = 1080 - 66
    largura = 1920 - 77
    figsize_init = (largura*px,altura*px)

plt.rcParams['axes.grid'] = True   #ativa grelha em todos os gráficos

# CORES DOS GRÁFICOS
c=['b','r','C2'] 
"====================================================="

"""Inputs cgds"""
star = "Kepler1705" #nome do Sistema
num_p = 2 #nº de planetas no sistema
Ct = 0.333 #momento principal de inércia [sem unidades]
kf = 0.9
G = 4*np.pi**2

# NOME DOS PLANETAS ===========================================================
planets = 'bcd'
#==============================================================================

"""****************************************************************************
                                VALORES DE TAU                              """
                                
# print(argv[1],len(argv[1]))
if argv[1] not in ['0','+0','+','p']:
    tau_list = list(map(float,argv[1].split(','))) #o utilizador define os valores de tau que quer

else:
    tau_list = [1E-3,1E-2,1E-1,1.] #lista default dos valores de tau
    
"""*************************************************************************"""


""" PASTAS DE DADOS *****************************************************************"""
#pastas dos vários conjuntos de dados possíveis
if argv[2] not in ['0','+0','+','p']:
    data_list = [argv[2]] #o utilizador define os dados que quer ler

else:
    data_list = ['e_HIGH','e_LOW'] #,'e_MED']

c22r_list = ['C22r = 0','C22r = 1E-7','C22r = 1.5E-6']

n_folder = 2
"""*************************************************************************"""

# ANGULOS =====================================================================
angle = 0 #theta inicial
#==============================================================================

# NIU DAS SOR =================================================================
#valor para fazer retas de ressoância;
niu = {'e_HIGH':0.00892,
       'e_LOW':0.00884,
       'e_MED':0.0089}
#==============================================================================

"""
Prints iniciais
"""
print('\nValores de tau:', tau_list)
print('Valores de angulos:',angle)
print('Dados: ', data_list)
print('C22r: ', c22r_list)
""""""

path = os.path.dirname(__file__)  # localização da pasta do sistema

"=========== PASTA PLOTS ================================"
mfolder = os.path.join(path,'PLOTS')
if not os.path.exists(mfolder) and 's' in argv: #criação de uma pasta mãe onde se guardam as pastas filhas
    os.mkdir(mfolder)
"========================================================"

#%% PASTA DOS GRÁFICOS, DADOS .csv DO SISTEMA

for data in data_list: # lê cada uma das pastas da lista data
    flush_print('\nDados:',data)

    "=========== PASTA PLOTS ================================"
    #criação de uma pasta filha (high,low,med) onde se guardam os plots
    plots_folder = os.path.join(mfolder,data)
    if not os.path.exists(plots_folder) and 's' in argv:
        os.mkdir(plots_folder)
    "========================================================"

    for tau in tau_list: #para cada valor de tau
        flush_print('\ntau=',tau)

        tau_name = '0.1E'+f'{tau*10:.1E}'[4:7]       #str de tau no formato de saída

        # for angle in angle_list: #para cada valor de theta inicial
            
        #=== FIGURA (UMA POR TAU) =================================================================================================

        fig, axs = plt.subplots(6, 3, figsize=figsize_init, 
                                sharex='all',sharey='row',layout='constrained')
        # fig.subplots_adjust(hspace=0.08)

        #========================================================================================================================
        
        for c22r in c22r_list:
            flush_print(' ',c22r)

            #=========== DIRECTORIA DOS FICHEIROS ======================================
            data_name = os.path.join( path+f'/n_planet = {n_folder}'+f'/{c22r}/',data )
            #===========================================================================

            #***** ARRAYS DE DADOS (para guardar os cálculos) *****#
            a_list=[0,0]
            #*******************************************************

            "FICHEIRO DE DADOS DO PLANETA"
            if tau == tau_list[0] and c22r == c22r_list[0]: #só é preciso fazer uma vez a leitura dos dados
                
                #***** ARRAYS DE DADOS (para guardar os cálculos) *****#
                mass_p = np.array([])
                R_p = np.array([])
                miu = np.array([])
                #*******************************************************

                " LEITURA "
                Planet_data = pd.read_csv(
                                            os.path.join(data_name,f'{star}.csv'),
                                            sep=r",",nrows=1,
                                            names=[i for i in range(0,6*num_p+2)],
                                            engine="python"
                                                                                        )         
                mass_0 = list(Planet_data.iloc[0])[-2] # massa da estrela em [M_sun]
                R_0 = list(Planet_data.iloc[0])[-1] # raio da estrela em [R_sun]
                for n in range(1,num_p+1): 
                    mass_p = np.append(mass_p,Planet_data.iloc[0][5*n-1]) #massas dos planetas, [M_star]
                    R_p = np.append(R_p,Planet_data.iloc[0][5*num_p-1+n]) #raios dos planetas, [R_star]

                mass_p = mass_p*mass_0  #conversão de M_star para M_sun    
                miu = G*(mass_0+mass_p)
                R_p = R_p*R_0 #conversão de R_star para R_sun
                R_p = R_p*695700.0/1.4959787E8 #conversão de R_sun para km e depois para AU
                Cmain = Ct*mass_p*R_p**2
        
#%% DADOS .car, .rot e CÁLCULOS ... ============================================================================================

            for p in range(len(planets[:num_p])): #para cada planeta

                filename = os.path.join(f'tau_=_{tau_name}', f'{angle}_{star}.{planets[p]}') #nome da pasta e dos ficheiros que lê
                filename = os.path.join(data_name,filename) #total de path + filename
                
                "FICHEIRO DE COORDENADAS: X.car"
                coord = pd.read_csv(
                                        filename+'.car',names=['t','x','y','vx','vy'],
                                        usecols=[0,1,2,4,5], sep=' ',                #ignora t, z e vz
                                        engine='python', skipinitialspace=True
                                                                                                        )
    
                # ******************************************************************************************
                #                               CÁLCULOS
                # ******************************************************************************************

                r_mod = (coord['x']**2 + coord['y']**2)**0.5 #módulo do vector posição r

                h = coord['x']*coord['vy'] - coord['y']*coord['vx']  #momento angular especifico (G1)
                
                E = 0.5*(coord['vx']**2+coord['vy']**2) - miu[p]/r_mod #Energia orbital específica
                

                a = -miu[p]/(2*E[1]) #semi-eixo
                # print(float(c22r[7:len(c22r)]))
                # miu1 = miu[p]
                miu[p] = miu[p] * ( 1 + 9*(0 + float(c22r[7:len(c22r)]))*(R_p[p]/a)**2 ) #correção do miu devido à precessão
                # print(miu[p]-miu1)
                
                a = -miu[p]/(2*E) #semi-eixo com correção do miu

                e = (1-h**2/(a*miu[p]))**0.5              #excentricidade

                # L = pd.DataFrame(data={'x':[], 'y':[]})
                # L['x'] = coord['vy']*h/miu[p] - coord['x']/r_mod
                # L['y'] = -coord['vx']*h/miu[p] - coord['y']/r_mod 

                # e = (L['x']**2+L['y']**2)**0.5
                # a = h**2/(miu[p]*(1-e**2))

                a_list[p] = a #guarda as excentricidade numa lista
                

                if p == 0:

                    e0 = e[0] #guarda primeiro valor de e_b para cálculos futuros

                    "FICHEIRO DE ROTAÇÃO E DEFORMAÇÃO: X.rot"
                    spin = pd.read_csv(
                                            filename+'.rot',names=['t','theta','Omega','J2','C22','S22'],
                                            sep=' ', engine='python',skipinitialspace=True                       
                                                                                                            )
                    
                    n = (miu[p]/a**3)**0.5
                    n0 = n[0]
                    
                    #Energia de rotação dissipada
                    Erot = 0.5*Cmain[p]*spin['Omega']**2
                    
                    #Energia orbital dissipada
                    Eorb = mass_0*mass_p[p]/(mass_0+mass_p[p])*E
                    # print(len(coord),len(E),len(a),len(Eorb))
                    
                    # ENERGIA TOTAL
                    Etot = - Eorb[:len(Erot)] - Erot[:len(Eorb)] #[m_sun AU^2/yr^2]
                    Etot = Etot/(4*np.pi*R_p[0]**2) #Energia por unidade de Área [m_sun Au^2 yr^-2 AU^-2]
                    Etot = Etot*1988400E24/(365.25*24*3600)**2 #conversão para [J/m^2] (J=kg m^2/s^2)

                    if len(coord)<len(spin):
                        time = coord['t']
                    else:
                        time = spin['t']
                        
                    #DERIVADAS DA ENERGIA
                    # dEorb = np.gradient(Eorb , dt) 
                    # dErot = Cmain[p]*spin['Omega']*np.gradient(spin['Omega'], dt)

                    #ENERGIA DISSIPADA POR MARÉ ESPERADA DE ACORDO COM HENNING 2009, A DIVIDIR PELA ÁREA TOTAL DO PLANETA (dEtot/A)
                    # Qmain = 20 #valor intermédio da Terra
                    # dEtotQ = 21/2*kf/Qmain*np.pi*mass_0**2*R_p[0]**3*miu[0]**0.5*e**2/a**(15/2) #[m_sun yr^-3 m^-2]
                    # dEtotQ = dEtotQ * 1988400E24 / (365.25*24*3600)**3 #conversão para [W/m^2]

                    #**********************************************************************************************
                    
                """
                DELETES
                """
                del(h,E,r_mod)


#%% PLOTS - MATPLOTLIB

                # TÍTULO
                # fig.suptitle(
                #                 f'{data}'+fr', $\tau$ = {tau}'+'\n'+fr'$\tau n$ = {tau*n0:.2E}, $\tau \nu$ = {tau*niu[data]*n0:.2E}',
                #                 fontweight="bold"
                # )

                fig.suptitle(
                fr'$\tau$ = {tau}, '+fr'$\tau n$ = {tau*n0:.2E}, $\tau \nu$ = {tau*niu[data]*n0:.2E}',
                fontweight="bold"
)
                
                xticks = [i/10*10**9 for i in range(0,11,1)]
                xlabels = [f'{i/10}' for i in range(0,11,1)] #etiquetas do eixo x
            
                for j in range(len(axs[0])):
                    # axs[len(axs)-1][j].set_xlabel('t [yr]')        #variável do eixo x

                    axs[len(axs)-1][j].xaxis.set_major_locator(ticker.MaxNLocator(11))
                    axs[len(axs)-1][j].set_xlabel('t [Gyr]')     #variável do eixo x
                    
                    # axs[len(axs)-1][j].set_xticks(xticks,labels=xlabels)
            
                indice = c22r_list.index(c22r) #forma cgd de fazer os graficos itrativamente 
                
                "__________________________________ EXCENTRICIDADE __________________________________"
                ax=axs[1][indice]
                ax.set_yscale('log')
                if tau == 0.001: 
                    ax.set_yticks([1E-4,1E-3,1E-2])
                else:
                    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=4)) 
                ax.minorticks_off()
                if indice == 0 : 
                    ax.set_ylabel(r'$e$')

                ax.plot((coord['t']/1e9),e,color=c[p],label=r'$\text{%s}$' % planets[p],zorder=4-p, alpha = 1-0.3*(1-p))       #para ficar por cima da grelha tem de ser >= 3
                if tau != 1: 
                    ax.legend(loc='upper right',ncols=2)
                else:
                    ax.legend(loc='lower right',ncols=2)


            "__________________________________ PERIODO: P2/P1  __________________________________"
            ax=axs[0][indice]
            ax.set_title(f'{c22r}')
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            if indice == 0 : ax.set_ylabel(r'$P_\text{c}~/~P_\text{b}$')
            if data == 'e_HIGH':
                if tau == 0.001 or tau == 0.01:
                    ax.set_yticks([1.250,1.255,1.260])
                # else:
                #     ax.set_yticks([1.25,1.50,1.75,2.0])

            ax.plot((coord['t']/1e9),(a_list[1]/a_list[0])**(3/2),color=c[0],zorder=4-p)
            ax.scatter((coord['t']/1e9)[0],((a_list[1]/a_list[0])**(3/2))[0],color='black',zorder=5-p)
            # ax.plot(coord['t'],[1.25 for i in range(0,len(coord['t']))], linestyle='--', color='grey', zorder = 6)
            
            # VALORES FINAIS DO RÁCIO DE PERIODOS
            # print(c22r,'\n-> Pc/Pb = ',((a_list[1]/a_list[0])**(3/2)).iloc[-1])

            # # GRÁFICO DAS EXCENTRICIDADES EM SEPARADO
            # ax=axs[5][indice]
            # ax.plot(time/1e9,a_list[1][:len(time)],c='r')
            # ax.plot(time/1e9,a_list[0][:len(time)],c='b')
            
            "__________________________________ OMEGA/N __________________________________"
            O_n = spin['Omega']/n[:len(spin)]
            # print(O_n[0])
            
            plot_pos=2
            ax=axs[plot_pos][indice]
            ax.set_ylim(1.1,2.5)
            ax.set_yticks([1.5,2,2.5])

            #ajustes nos eixos para quebra do eixo-y
            #   parametros para quebra do eixo-y 
            d = 0.25
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                            linestyle="none", color='k', mec='k', mew=1, clip_on=False)

            ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)
            ax.spines.bottom.set_visible(False)
            ax.tick_params(
                            axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off

            ax.plot((spin['t']/1e9),O_n,color = c[0],label=r'$\Omega$/n',
                    linestyle='dashed',zorder=4)
            ax.legend(loc='lower right')
            

            #'	__________________________________ Zoom in O/n ____________________________________________'
            ax=axs[plot_pos+1][indice]
            ax.set_ylim(1-2*niu[data],1+2*niu[data])

            ax.spines.top.set_visible(False)
            ax.plot([0, 1], [1, 1], transform=ax.transAxes, **kwargs) #plot da quebra do eixo-y

            ax.plot((spin['t']/1e9),O_n,color = c[0],
                    linestyle='dashed',zorder=3)

            'linhas de ressonância'
            points = 1
            ax.plot(
            (spin['t']/1e9)[::points],np.ones(len(spin['t']/1e9))[::points]       ,'limegreen',
            (spin['t']/1e9)[::points],[1+niu[data]/2   for j in spin['t']/1e9][::points],'limegreen',
            (spin['t']/1e9)[::points],[1-niu[data]/2   for j in spin['t']/1e9][::points],'limegreen',
            (spin['t']/1e9)[::points],[1+niu[data]     for j in spin['t']/1e9][::points],'limegreen',
            (spin['t']/1e9)[::points],[1-niu[data]     for j in spin['t']/1e9][::points],'limegreen',
            (spin['t']/1e9)[::points],[1+6*e0**2 for j in spin['t']/1e9][::points],'green'            
                                                                                                )


            "__________________________________ ENERGIA DISSPADA __________________________________"
            # if len(coord)<len(spin):
            #     time = coord['t']
            # else:
            #     time = spin['t']
            
            
            ##---> ENERGIA TOTAL
            ax=axs[4][indice]
            # ax.set_yscale('log')
            if data == 'e_HIGH':
                ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
                if tau == 0.1 or tau == 1.0:
                    ax.set_yticks([7.8,8.4,9.0])
                elif tau == 0.01:
                    ax.set_ylim(7.59,7.67)
                    ax.set_yticks([7.6,7.62,7.64,7.66])
            elif data == 'e_LOW':
                ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
                if tau == 0.001:
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            if indice == 0: 
                ax.set_ylabel(r'$E_{tot}~/~A$'+'\n'+r'$\left[\times 10^{19}~J/m^2\right]$')

            ax.plot((time[::]/1e9), Etot[:len(time):]/1E19, zorder=3, color='xkcd:blood red')


            #---> AJUSTE POLINOMIAL DA ENERGIA
            # np.plyfit calcula os coefecientes e np.poly1d calcula os valores 

            ti = 20000 #corresponde a ~2E8 yr
            if data == 'e_HIGH':
                if tau in [0.001,0.01]:
                    grau = 1
                elif tau == 0.1:
                    grau = 12
                elif tau == 1. and '6' in c22r:
                    grau = 2
                elif tau == 1.:
                    grau = 1
            if data == 'e_LOW':
                if tau in [0.001,0.01]:
                    grau = 2
                elif tau == 0.1:
                    grau = 5
                elif tau == 1.:
                    if '6' in c22r:
                        grau = 1
                    elif '0' in c22r:
                        grau = 4
                    elif '7' in c22r:
                        grau = 6
            # print(Etot[ti:len(time)])
            Etot_fit = np.poly1d(np.polyfit(time[ti:],Etot[ti:],grau))(time[ti:])

            # ax.plot(time[ti:]/1e9,Etot_fit/1E19,zorder=4,label=r'fit', c='C1')
            # ax.legend(loc='lower right')


            ##---> DERIVADA DO AJUSTE POLINOMIAL DA ENERGIA TOTAL
            ax=axs[5][indice]
            if indice == 0: ax.set_ylabel(r'$d_t~(E_{tot}~/~A)$'+'\n'+r'$\left[W/m^2\right]$')

            dEtot = np.gradient(Etot_fit,time[1]*(365.25*24*3600)) # W m^-2
            # ax.plot(time[ti:],dEtot,zorder=3,label=r'$\frac{d}{dt}(E_{tot,fit})$', c = 'C1')

            if data == 'e_HIGH':
                window = 2500
                for i in range(1):
                    dEtot = movingaverage(dEtot,window)
            elif data == 'e_LOW':
                if tau >= 0.1:
                    window = 2500
                    for i in range(1):
                        dEtot = movingaverage(dEtot,window)
                else:
                    window = 12
                    for i in range(1,11,3):
                        dEtot = movingaverage(dEtot,window*(i**2))


            ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
            if data == 'e_HIGH':
                if tau == 0.001:
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
                    ax.set_ylim(-0.5,max(dEtot)*(1.5))
                    # ax.set_yticks([0,1.5,3])
                elif tau == 0.01:
                    ax.set_ylim(-0.5,max(dEtot)*(5))      
                elif tau in [0.1,1]:
                    # ax.set_yticks([0,50,100,150])
                    ax.set_ylim(-10,160)
            elif data == 'e_LOW':
                if tau in [0.001,0.01]:
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
                    ax.set_ylim(-0.5,max(dEtot)*(1.5))


            # ax.plot((time[ti:len(time)-window+1]/1e9),dEtot,zorder=3, c = 'C0')
            ax.plot(time[ti:len(dEtot)+ti]/1e9,dEtot,zorder=3, c = 'C0')


            # # "_____________________________________________________ OFFSET _____________________________________________________"
            # window = 100 
            # a_mean_list = [movingaverage(a,window) for a in a_list]
            # Delta_54 =  a_mean_list[1][-1]**(3/2)/a_mean_list[0][-1]**(3/2) - a_mean_list[1][ti]**(3/2)/a_mean_list[0][ti]**(3/2)

            # #  MILLHOLLAND 2019
            # Delta_obs = 0.05
            # Delta_obs54  = 5/4*(Delta_obs +1) - 5/4

            # # #  OBSERVAÇÕES
            # # Delta_obs32 = P_p[1]/P_p[0] - 3/2 #offset observado nos dados

            # t_evol54 = Delta_obs54/Delta_54 * coord['t'].iloc[-1]
            # if '0' in c22r: print(f'tau = {tau}')
            # print('->',c22r, f', t(5:4) = {t_evol54:.6e}')
            # # print('5:4 - ',Delta_obs54,Delta_54,coord['t'].iloc[-1])
            
            # ax=axs[0][indice]
            # ax.plot([(coord['t']/1e9)[ti],(coord['t']/1e9).iloc[-1]],[((a_mean_list[1]/a_mean_list[0])**(3/2))[ti],((a_mean_list[1]/a_mean_list[0])**(3/2))[-1]]
            #         , color='gold',zorder=5,marker='o')


            #  HENNING 2009
            # ax=axs[6][indice]
            # ax.plot(coord['t'][ti:], dEtotQ[ti:] ,label = r'$\dot{E}~(Henning ~2009)$',c='C2')
            # ax.legend(loc='upper right')

            # window = 700 
            # dEtot_mean = movingaverage(dEtot,window)
            # dEtot_mask = dEtot[:len(dEtot_mean)] > 1.5*dEtot_mean
            # print(dEtot_mask)
            # for i in range(len(dEtot_mask)):
            #     if dEtot_mask[i] == False:
            #         dEtot_mask[i] = dEtot[i]
            #     else:
            #         dEtot_mask[i] = dEtot_mean[i]

            # ax.plot(time[ti:len(time)-window+1],dEtot_mask,zorder=4, c = 'C0')


        "====== GUARDAR FIGURAS ========================================================"
        if 'p' in argv: #se o utilizador puser opção de ver plots interativos
            plt.show()
        elif 's' in argv: #se o utilizador puser opção de guardar plots
            plt.savefig(os.path.join(plots_folder,f'tau={tau}_{angle}.png'),dpi=500)
            plt.close('all')
        else:
            plt.close('all')

        flush_print(f'FINISHED, t={tau}, a={angle}º\n')


# flush_print('THE END\n')
