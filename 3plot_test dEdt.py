# -*- coding: utf-8 -*-
"""
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
    
@author: José Rodrigues, 19/6/2024
"""

# %% IMPORTS

# argv = ['','0','0','0','p']
from sys import argv
print('\nargv: ', argv,'\n')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def movingaverage(data, window_width):
    """
    Computes moving average of an array
    """
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width


def flush_print (*args):
    if argv[1][0] == '+':
        import datetime
        now = datetime.datetime.now()
        timestamp = now.strftime('%H:%M:%S.%f')
        print (timestamp,  *args, flush=True)
        return
    
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

flush_print('Initiating...\n')

"TAMANHO DAS IMAGENS"
# altura  = 1080
# largura = 1920

altura = 718
largura = 1420

altura  = altura - 66
largura = largura - 70

#%% INPUTS

"""Inputs cgds"""
star = "Kepler1705" #nome do Sistema
num_p = 2 #nº de planetas no sistema
Ct = 0.333 #momento principal de inércia [sem unidades]

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

if argv[3] not in ['0','+0','+','p']:
    c22r_list = ['C22r = '+ argv[3]]

else:
    c22r_list = ['C22r = 1E-7']
c22r_list.append('C22r = 0')
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
mfolder = os.path.join(path,'PLOTS2')
if not os.path.exists(mfolder) and 's' in argv: #criação de uma pasta mãe onde se guardam as pastas filhas
    os.mkdir(mfolder)

#criação de uma pasta filha (c22r) onde se guardam os plots
c22r_folder = os.path.join(mfolder,c22r_list[0])
if not os.path.exists(c22r_folder) and 's' in argv:     
    os.mkdir(c22r_folder)
"========================================================"

#%% PASTA DOS GRÁFICOS, DADOS .csv DO SISTEMA

for data in data_list: # lê cada uma das pastas da lista data
    flush_print('\nDados:',data)

    "=========== PASTA PLOTS ================================"
    #criação de uma pasta filha (high,low,med) onde se guardam os plots
    plots_folder = os.path.join(c22r_folder,data)
    if not os.path.exists(plots_folder) and 's' in argv:
        os.mkdir(plots_folder)
    "========================================================"

    for tau in tau_list: #para cada valor de tau
        flush_print('\ntau=',tau)

        tau_name = '0.1E'+f'{tau*10:.1E}'[4:7]       #str de tau no formato de saída

        # for angle in angle_list: #para cada valor de theta inicial
            
        #=== FIGURA (UMA POR TAU) =================================================================================================

        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        plt.rcParams['axes.grid'] = True   #ativa grelha em todos os gráficos

        fig, axs = plt.subplots(2, 3, figsize=(largura*px,altura*px), 
                                sharex='all',sharey='row',layout='constrained')
        # fig.subplots_adjust(hspace=0.08)

        #========================================================================================================================
        
        for n_folder in range(1,3):
            flush_print(' Nº planetas =',n_folder)

            for c22r in c22r_list[0:n_folder]: #abre apenas a pasta 1E-7 quando n_folder = 1, e abre as duas pastas quando n_folder = 2
                flush_print(' ',c22r)

                #=========== DIRECTORIA DOS FICHEIROS ======================================
                data_name = os.path.join( path+f'/n_planet = {n_folder}'+f'/{c22r}/',data )
                #===========================================================================

                "FICHEIRO DE DADOS DO PLANETA"
                if tau == tau_list[0] and n_folder == 1 and c22r == c22r_list[0]: #só é preciso fazer uma vez a leitura dos dados
                    
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

                    mass_p = mass_p*mass_0    
                    miu = G*(mass_0+mass_p)
                    R_p = R_p*R_0
                    Cmain = Ct*mass_p*R_p**2
            
#%% DADOS .car, .rot e CÁLCULOS ... ============================================================================================

                # for p in range(len(planets[:num_p])): #para cada planeta

                p = 0

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

                h = coord['x']*coord['vy'] - coord['y']*coord['vx']  #momento angular especifico
                
                E = 0.5*(coord['vx']**2+coord['vy']**2) - miu[p]/r_mod #Energia orbital específica
                 

                a = -miu[p]/(2*E)                         #semi-eixo
                e = (1-h**2/(a*miu[p]))**0.5              #excentricidade

                # if planets[p] == 'b':

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

                #DERIVADAS DA ENERGIA
                # dEorb = np.gradient(Eorb , dt) 
                # dErot = Cmain[p]*spin['Omega']*np.gradient(spin['Omega'], dt)

                #**********************************************************************************************
                
                """
                DELETES
                """
                del(h,E,r_mod)


#%% PLOTS - MATPLOTLIB

                c='bry' #cores dos gráficos

                # TÍTULO
                fig.suptitle(
                                f'{data}'+fr', $\tau$ = {tau}'+'\n'+fr'$\tau n$ = {tau*n0:.2E}, $\tau \nu$ = {tau*niu[data]*n0:.2E}',
                                fontweight="bold"
                )
                
                xticks = [i*10**6 for i in range(0,11,1)]
                xlabels = [f'{i}' for i in range(0,11,1)] #etiquetas do eixo x

                for j in range(len(axs[0])):
                    axs[len(axs)-1][j].set_xlabel('t [yr]')        #variável do eixo x
                    # axs[len(axs)-1].set_xlabel('t [Myr]')     #variável do eixo x
                    # axs[j].set_xticks(xticks,labels=xlabels)
                    # axs[j].set_xscale('log')
                    # axs[j].set_xticks(xticks) 
                    # axs[j].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            
                indice = n_folder-1+c22r_list.index(c22r) #forma cgd de fazer os graficos itrativamente
                

            #     "__________________________________ SEMI-EIXO __________________________________"
            #     ax=axs[0][indice]
            #     ax.set_title(f'{c22r}, n = {n_folder}')
            #     ax.plot(coord['t'],a,color=c[p],label=f'a$_{planets[p]}$ [AU]',zorder=4-p) 
            #     ax.legend(loc='upper right')
                
            #     "__________________________________ EXCENTRICIDADE __________________________________"
            #     ax=axs[1][indice]
            #     ax.set_yscale('log')
            #     ax.plot(coord['t'],e,color=c[p],label=f'e$_{planets[p]}$',zorder=4-p)       #para ficar por cima da grelha tem de ser >= 3
            #     ax.legend(loc='upper right')
                
                
            #     "__________________________________ OMEGA/N __________________________________"
                
            #     #parametros para quebra do eixo-y
            #     d = 0.25
            #     kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
            #                     linestyle="none", color='k', mec='k', mew=1, clip_on=False)

            #     O_n = spin['Omega']/n[:len(spin)]
            #     # print(O_n[0])
                
            #     plot_pos=2
            #     ax=axs[plot_pos][indice]

            #     #ajustes nos eixos para quebra do eixo-y
            #     ax.spines.bottom.set_visible(False)
            #     ax.tick_params(
            #                     axis='x',          # changes apply to the x-axis
            #                     which='both',      # both major and minor ticks are affected
            #                     bottom=False,      # ticks along the bottom edge are off
            #                     top=False,         # ticks along the top edge are off
            #                     labelbottom=False) # labels along the bottom edge are off

            #     ax.set_ylim(1.1,2.5)
            #     ax.set_yticks([1.5,2,2.5])
            #     ax.plot(spin['t'],O_n,color='b',label=r'$\Omega$/n',
            #             linestyle='dashed',zorder=4)
            #     points = 1
            #     ax.plot(spin['t'][::points],[1.5 for j in spin['t']][::points],'green',
            #             spin['t'][::points],np.ones(len(spin['t']))[::points],'limegreen',)
                
            #     #plot da quebra do eixo-y
            #     ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)

            #     ax.legend(loc='upper right')
                
            #     # if tau < 10.:

            #    #'	__________________________________ Zoom in O/n ____________________________________________'
            #     ax=axs[plot_pos+1][indice]
            #     ax.spines.top.set_visible(False)
            #     # ax.set_title('Zoom-in of Resonance Region')
            #     ax.set_ylim(1-2*niu[data],1+2*niu[data])
                
            #     # if tau == 0.1: ax.set_ylim(0.8,1.2)
                
            #     # ax.set_yticks([1-2*niu[data],1,1+2*niu[data]]) 

            #     ax.plot(spin['t'],O_n,color='b',
            #             linestyle='dashed',zorder=3)

            #     'linhas de ressonância'
            #     if n_folder == 1:
            #         ax.plot(spin['t'][::points],np.ones(len(spin['t']))[::points],'limegreen')
            #     else:
            #         ax.plot(
            #         spin['t'][::points],np.ones(len(spin['t']))[::points]       ,'limegreen',
            #         spin['t'][::points],[1+niu[data]/2   for j in spin['t']][::points],'limegreen',
            #         spin['t'][::points],[1-niu[data]/2   for j in spin['t']][::points],'limegreen',
            #         spin['t'][::points],[1+niu[data]     for j in spin['t']][::points],'limegreen',
            #         spin['t'][::points],[1-niu[data]     for j in spin['t']][::points],'limegreen',
            #         spin['t'][::points],[1+6*e0**2 for j in spin['t']][::points],'green'            
            #                                                                                             )
                
            #     #plot da quebra do eixo-y
            #     ax.plot([0, 1], [1, 1], transform=ax.transAxes, **kwargs)


                "__________________________________ ENERGIA DISSPADA __________________________________"

                if len(coord)<len(spin):
                    time = coord['t']
                else:
                    time = spin['t']

                ##---> ENERGIA TOTAL
                ax=axs[0][indice]

                Etot = - Eorb[:len(time)] - Erot[:len(time)]
                if indice == 0: ax.set_ylabel(r'$E_{tot}$')
                ax.plot(time[::], Etot[:len(time):], zorder=3, label = r'$E_{tot}$', color='xkcd:blood red')
                # ax.legend(loc='upper right')


                #---> AJUSTE POLINOMIAL DA ENERGIA
                # np.plyfit calcula os coefecientes e np.poly1d calcula os valores 

                ti = 10000 #corresponde a 1E8 yr
                for i in [9,13]:
                    Etot_fit = np.poly1d(np.polyfit(time[ti:],Etot[ti:],i))(time[ti:])
                    ax.plot(time[ti:],Etot_fit,zorder=4,label=r'$E_{tot,fit}$'+f' {i}')
                    ax.legend(loc='best')


                ##---> DERIVADA DA MÉDIA ENERGIA TOTAL
                ax=axs[1][indice]

                dEtot = np.gradient(Etot_fit,time[1])
                ax.plot(time[ti:],dEtot,zorder=3,label=r'$\frac{d}{dt}(E_{tot,fit})$')
                ax.legend(loc='best')

                # step = 1000
                # #dt = int(time[step]) #passo entre dados
                # dt = int(time[1])
                # dEtot = np.gradient(Etot_mean , dt) 
                # ax=axs[5][indice]
                # if indice == 0: ax.set_ylabel(r'$\dot{\langle E_{tot} \rangle}$')
                # ax.plot(time[:len(dEtot)],  dEtot[:len(time)], zorder=3, label = r'$\dot{E}_{tot}$', color='C1')

                # dt = int(coord['t'][step*50]) #passo entre dados

                # dEtot = np.gradient(Etot_mean[::step] , dt) 
                # ax=axs[5][indice]
                # if indice == 0: ax.set_ylabel(r'$\dot{\langle E_{tot} \rangle}$')
                # ax.plot(time[::step],  dEtot, zorder=3, label = r'$\dot{E}_{tot}$', color='C1')


                # dEtot = derivative(Etot_mean , dt) 
                # ax=axs[5][indice]
                # ax.plot(time[:len(dEtot):],  dEtot[:len(time):], zorder=3, label = r'$\dot{E}_{tot}: mein$', color='xkcd:blood red')
                # ax.legend(loc='upper right')
                


                ##---> MÉDIA MÓVEL DA ENERGIA TOTAL

                # max_time = 10000 #corresponde a 100 E6 yr
                # Etot_mean1 = movingaverage(Etot[:max_time:1], 100)
                # Etot_mean2 = movingaverage(Etot[max_time:len(time):1],100)
                # for i in range(100):
                #     Etot_mean2 = movingaverage(Etot_mean2,10)
                # Etot_mean = np.concatenate((Etot_mean1,Etot_mean2))

                # Etot_mean = movingaverage(Etot,70)
                # # for i in range(50):
                # #     Etot_mean = movingaverage(Etot_mean,50)

                # ax.plot(time[:len(Etot_mean)], Etot_mean[:len(time)], zorder=4,
                # label=r'$\langle E_{tot} \rangle$',color='red')
                # # ax.legend(loc='best')

                # step = 1
                # Etot_mean = Etot.rolling(window=step).mean()
                # while step < 20:
                #     step += 1
                #     Etot_mean = Etot_mean.rolling(window=step).mean()
                # ax.plot(time[step::], Etot_mean[step::], zorder=4,
                # label=r'$\langle E_{tot} \rangle$',color='red')
                # ax.legend(loc='best')

                #print(time[::step].head(),dEtot_mean[:len(time):step].head())
                # ax=axs[5][indice]
                # if indice == 0: ax.set_ylabel(r'$\langle E_{tot} \rangle$')



                ##---> ENERGIA DE ROTAÇÃO
                # ax=axs[6][indice]
                # ax.plot(time, Erot[:len(time)], zorder=5,linestyle='dashed', label = r'$E_{rot}$')
                # ax.legend(loc='upper right')

                ##---> ENERGIA ORBITAL
                # ax=axs[5][indice]
                # ax.plot(time, Eorb[:len(time)], zorder=4,linestyle='dashed', label = r'$E_{orb}$',c='C1')
                # ax.legend(loc='upper right')

                # del(time,Etot_mean1,Etot_mean2)
                

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
