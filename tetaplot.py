# -*- coding: utf-8 -*-
"""
PLOTS EM PYTHON

- OS CALCULOS SÃO TODOS PARA O CASO 2D (z=0 e vz=0)

- O PLANETA DEFORMADO É O PLANETA b 

- PRECISA DOS FICHEIROS .car .rot e *ParamInit.out*
de uma lista dos valores de tau, de uma lista dos ângulos e
de uma lista da pasta dos dados

Para correr o programa:
    python3 allpyplot.py (+)*valores de tau* *dados* *c22r* *c/ ou s/ plots interativos*
    
    Ex.: python3 allpyplot.py +1E-1 e_HIGH 0 p
    
        ->  +: faz prints de debug (sem debug não colocar +)
        ->  1E-1: valor de tau utilizado (separar vários valores por virgulas)
        -> e_HIGH: lê os dados da pasta e_HIGH
        -> 0: todos os valores de c22r
        ->  p: c/ gráficos interativos, mas não os guarda (sem plots colocar 0, para os guardar na pasta)

Para correr gravando apenas os graficos numa pasta:
    python3 allpyplot.py 0 0 0
    
@author: José Rodrigues, 22/5/2024
"""

#%% IMPORTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sys import argv

def flush_print (*args):
    if argv[1][0] == '+':
        import datetime
        now = datetime.datetime.now()
        timestamp = now.strftime('%H:%M:%S.%f')
        print (timestamp,  *args, flush=True)
        return

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

# NOME DOS PLANETAS ===========================================================
planets = 'bcd'

c='bry' #cores dos gráficos
#==============================================================================


""" PASTAS *****************************************************************"""
#pastas dos vários conjuntos de dados possíveis
if argv[2] not in ['0','+0','+','p']:
    data_list = [argv[2]] #o utilizador define os dados que quer ler

else:
    data_list = ['e_HIGH','e_LOW']#'e_MED']

if argv[3] not in ['0','+0','+','p']:
    c22r_list = ['C22r = ' + argv[3]] #o utilizador define os dados que quer ler

else:
    c22r_list = ['C22r = 1.5E-6','C22r = 1E-7']
    
"""*************************************************************************"""

# NIU DAS SOR =================================================================
#valor para fazer retas de ressoância;
niu = {'e_HIGH':0.00892,
       'e_LOW':0.00884,
       'e_MED':0.0089}
#==============================================================================

"""
Prints iniciais
"""
print('\nC22r: ', c22r_list,'\n')
""""""

path = os.path.dirname(__file__)  # localização da pasta do sistema

"=========== PASTA PLOTS ================================"
mfolder = os.path.join(path,'PLOTS_theta+')
if not os.path.exists(mfolder) and 's' in argv: #criação de uma pasta mãe onde se guardam as pastas filhas
    os.mkdir(mfolder)
"========================================================"


#%% DADOS .csv DO SISTEMA
for c22r in c22r_list:

    """****************************************************************************
                                VALORES DE TAU                              """
                                
    # print(argv[1],len(argv[1]))
    if argv[1] not in ['0','+0','+','p']:
        tau_list = list(map(float,argv[1].split(','))) #o utilizador define os valores de tau que quer

    else:
        if c22r == 'C22r = 1.5E-6':
            tau_list = [1E-3,1E-2,1E-1,1.] #lista default dos valores de tau
        else:
            tau_list = [1E-3,1E-2,1E-1,1.,10.,100.,1000.] 

    print('\nValores de tau:', tau_list)
    """*************************************************************************"""


    # ANGULOS =====================================================================
    if c22r == 'C22r = 1.5E-6':
        theta_max = 4
    else:
        theta_max = 9

    angle_list = [i for i in range(1,theta_max+1)] #theta

    print('Valores de angulos:',angle_list)
    #==============================================================================
    
    print('\n',c22r)

    "=========== PASTA PLOTS ================================"
    #criação de uma pasta filha (c22r) onde se guardam os plots
    c22r_folder = os.path.join(mfolder,c22r)
    if not os.path.exists(c22r_folder) and 's' in argv:
        os.mkdir(c22r_folder)
    "========================================================"

    #***** ARRAYS DE DADOS (para guardar os cálculos) *****
    mass_p = np.array([])
    miu = np.array([])
    R_p = np.array([])
    #*******************************************************
    
    for data in data_list: # lê cada uma das pastas da lista data
        flush_print('Pasta:',data)  

        "=========== PASTA PLOTS ================================"
        #criação de uma pasta filha (high,low,med) onde se guardam os plots
        plots_folder = os.path.join(c22r_folder,data)
        if not os.path.exists(plots_folder) and 's' in argv:
            os.mkdir(plots_folder)
        "========================================================"
        
        #=========== DIRECTORIA DOS FICHEIROS ======================================
        data_name = os.path.join(path+'/n_planet = 2'+f'/{c22r}/',data)
        #===========================================================================
        
        "FICHEIRO DE DADOS DO PLANETA"
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
        miu = 4*np.pi**2*(mass_0+mass_p)
        R_p = R_p*R_0
        Cmain = Ct*mass_p*R_p**2 
        
        """
        DELETES
        """
        # del(Planet_data,n,mass,mass_0)
            
    
    #%% DADOS .car, .rot e CÁLCULOS ... ============================================================================================
    
        for tau in tau_list: #para cada valor de tau
            flush_print('tau=',tau)

            tau_name = '0.1E'+f'{tau*10:.1E}'[4:7]       #str de tau no formato de saída


            if tau > 1.:
                angle_list = [i for i in range(0,11)] #theta

            for angle in angle_list: #para cada valor de theta
                flush_print('anlge=',angle,'º')

                
            #=== FIGURA (UMA POR THETA) =================================================================================================
    
                px = 1/plt.rcParams['figure.dpi']  # pixel in inches
                
                # if tau < 10. :
                fig, axs = plt.subplots(5, 1, figsize=(largura*px,altura*px), 
                                        layout='constrained',sharex=True
                                                                                )
                # else:
                #     fig, axs = plt.subplots(num_p+3, 1, figsize=(largura*px,altura*px), 
                #             layout='constrained',sharex=True
                #                                                                     )   
            #========================================================================================================================
                
                # for p in range(len(planets[:num_p])): #para cada planeta
                for p in range(1):
                    flush_print('planeta: ',planets[p])

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

                    dt = coord['t'][1]-coord['t'][0] #passo de escrita
                    
                    h = coord['x']*coord['vy'] - coord['y']*coord['vx']  #momento angular 
                    
                    E = (coord['vx']**2+coord['vy']**2)*0.5 - \
                        miu[p]/(coord['x']**2+coord['y']**2)**0.5 #Energia específica
                    

                    a = -miu[p]/(2*E)                         #semi-eixo
                    e = (1-h**2/(a*miu[p]))**0.5              #excentricidade

                    if planets[p] == 'b':
    
                        e0 = e[0] #guarda primeiro valor de e_b para cálculos futuros
    
                        "FICHEIRO DE ROTAÇÃO E DEFORMAÇÃO: X.rot"
                        spin = pd.read_csv(
                                            filename+'.rot',names=['t','Omega','J2','C22','S22'],
                                            usecols=[0,2,3,4,5],sep=' ',               #ignora theta
                                            engine='python',skipinitialspace=True )
                        
                        # eps = (spin['C22']**2+spin['S22']**2)**0.5 #C22 e S22
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
    #%% PLOTS - MATPLOTLIB
                                    # TÍTULO
                    fig.suptitle(
                                f'{data}'+fr', $\tau$ = {tau}'+'\n'+fr'$\tau n$ = {tau*n0:.2E}, $\tau \nu$ = {tau*niu[data]*n0:.2E}',
                                fontweight="bold"
                )
                    axs[0].set_title(f'{c22r}, ' + r'$\theta=$'+f'{angle}')     #TÍTULO
                    axs[len(axs)-1].set_xlabel('t [yr]')        #variável do eixo x
                    # axs[len(axs)-1].set_xlabel('t [Myr]')     #variável do eixo x
                    xticks = [i*10**6 for i in range(0,11,1)]
                    xlabels = [f'{i}' for i in range(0,11,1)] #etiquetas do eixo x
                    for j in range(len(axs)):
                        axs[j].grid(True)
                        # axs[j].set_xticks(xticks,labels=xlabels)
                        # axs[j].set_xscale('log')
                        # axs[j].set_xticks(xticks) 
                        # axs[j].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                
                
                    "__________________________________ SEMI-EIXO __________________________________"
                    ax=axs[0]
                    ax.plot(coord['t'],a,color=c[p],label=f'a$_{planets[p]}$ [AU]',zorder=4-p) 
                    ax.legend(loc='upper right')
                    
                    "__________________________________ EXCENTRICIDADE __________________________________"
                    ax=axs[1]
                    ax.set_yscale('log')
                    ax.plot(coord['t'],e,color=c[p],label=f'e$_{planets[p]}$',zorder=4-p)       #para ficar por cima da grelha tem de ser >= 3
                    ax.legend(loc='upper right')
                    
                    
                "__________________________________ OMEGA/N __________________________________"
                
                #parametros para quebra do eixo-y
                d = 0.25
                kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                                linestyle="none", color='k', mec='k', mew=1, clip_on=False)

                O_n = spin['Omega']/n[:len(spin)]
                # print(O_n[0])
                
                plot_pos=2
                ax=axs[plot_pos]

                #ajustes nos eixos para quebra do eixo-y
                ax.spines.bottom.set_visible(False)
                ax.tick_params(
                                axis='x',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False) # labels along the bottom edge are off

                ax.set_ylim(1.1,2.5)
                ax.set_yticks([1.5,2,2.5])
                ax.plot(spin['t'],O_n,color='b',label=r'$\Omega$/n',
                        linestyle='dashed',zorder=4)
                points = 1
                ax.plot(spin['t'][::points],[1.5 for j in spin['t']][::points],'green',
                        spin['t'][::points],np.ones(len(spin['t']))[::points],'limegreen',)
                
                #plot da quebra do eixo-y
                ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)

                ax.legend(loc='upper right')
                
                # if tau < 10.:

               #'	__________________________________ Zoom in O/n ____________________________________________'
                ax=axs[plot_pos+1]
                ax.spines.top.set_visible(False)
                # ax.set_title('Zoom-in of Resonance Region')
                ax.set_ylim(1-2*niu[data],1+2*niu[data])
                
                # if tau == 0.1: ax.set_ylim(0.8,1.2)
                
                # ax.set_yticks([1-2*niu[data],1,1+2*niu[data]]) 

                ax.plot(spin['t'],O_n,color='b',
                        linestyle='dashed',zorder=3)

                'linhas de ressonância'
                ax.plot(
                spin['t'][::points],np.ones(len(spin['t']))[::points]       ,'limegreen',
                spin['t'][::points],[1+niu[data]/2   for j in spin['t']][::points],'limegreen',
                spin['t'][::points],[1-niu[data]/2   for j in spin['t']][::points],'limegreen',
                spin['t'][::points],[1+niu[data]     for j in spin['t']][::points],'limegreen',
                spin['t'][::points],[1-niu[data]     for j in spin['t']][::points],'limegreen',
                spin['t'][::points],[1+6*e0**2 for j in spin['t']][::points],'green'            
                                                                                                    )
                
                #plot da quebra do eixo-y
                ax.plot([0, 1], [1, 1], transform=ax.transAxes, **kwargs)
                    

                "__________________________________ ENERGIA DISSPADA __________________________________"
                if len(coord)<len(spin):
                    time = coord['t']
                else:
                    time = spin['t']

                ##---> ENERGIA TOTAL
                ax=axs[4]

                Etot = - Eorb[:len(time)] - Erot[:len(time)]
                ax.plot(time[::], Etot[:len(time):], zorder=3, label = r'$E_{tot}$', color='xkcd:blood red')
                ax.legend(loc='upper right')


                # #---> AJUSTE POLINOMIAL DA ENERGIA
                # # np.plyfit calcula os coefecientes e np.poly1d calcula os valores 

                # ti = 10000 #corresponde a 1E8 yr
                # Etot_fit = np.poly1d(np.polyfit(time[ti:],Etot[ti:],12))(time[ti:])
                # ax.plot(time[ti:],Etot_fit,zorder=4,label=r'$E_{tot,fit}$', c='r')
                # ax.legend(loc='best')


                # ##---> DERIVADA DA MÉDIA ENERGIA TOTAL
                # ax=axs[5]

                # dEtot = np.gradient(Etot_fit,time[1])
                # ax.plot(time[ti:],dEtot,zorder=3,label=r'$\frac{d}{dt}(E_{tot,fit})$')
                # ax.legend(loc='best')

                
                if 'p' in argv: #se o utilizador puser opção de ver plots interativos
                    plt.show()
                elif 's' in argv:
                    plt.savefig(os.path.join(plots_folder,f'tau={tau}_{angle}.png'),dpi=500)
                    plt.close('all')
                else:
                    plt.close('all')
                
                flush_print(f'FINISHED, t={tau}, a={angle}º\n')
    
    
# flush_print('THE END\n')


