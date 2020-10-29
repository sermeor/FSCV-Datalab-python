
from flask import Flask
from flask import request, jsonify
from sklearn.metrics import r2_score
from flask_cors import CORS
from scipy import signal
import scipy
from scipy.integrate import simps
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint
import math
import pandas as pd
import numpy as np
import pickle as pk





app = Flask(__name__)
CORS(app, supports_credentials=True)


app.config["DEBUG"] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
@app.route('/deconvolution', methods=['GET'])

def deconvolution():

    if 'b' in request.args and 'Ts' in request.args and 'D' in request.args and 'signal' in request.args:
        b = float(request.args['b'])
        Ts = float(request.args['Ts'])
        D = float(request.args['D'])
        s=str(request.args['signal']).split(" ")
        s= [float(i) for i in s]
        tau=601.2*(1/(D*10**(6)))*b-0.065 #Experimental equation
        A=b*(Ts/tau)
        t=list(range(len(s))) #Range of temporal samples.
        ERF=[A*math.exp(-i/tau) for i in t]
        #DECONVOLUTION OF MASS TRANSPORT
        zerofill=list(np.zeros(len(s)-1))
        szeros=s+zerofill

        try:
            DCsignal=list(signal.deconvolve(szeros,ERF))
            return " ".join(str(x) for x in DCsignal[0])

        except:
            return "Error"


    else:
        return "Error: No b (electrode absorption strength) field provided. Please specify a b value."




@app.route('/peaks', methods=['GET'])

def peaks():
    if 'sign' in request.args and 'signal' in request.args and 'period' in request.args and 'ytype' in request.args:
        s=str(request.args['signal']).split(" ")
        s= [float(i) for i in s]
        s2=str(request.args['sign'])
        ytype=str(request.args['ytype'])
        period=float(str(request.args['period']))
        x=np.linspace(0, (len(s)-1)*period, num=len(s))
        s_area=np.array(s)

        if s2 == "negative":
            s=[-i for i in s]
            s=np.array(s)
            mean_s=abs(np.mean(s))
            s_relative=s/mean_s
            prom=0.1
        elif ytype == "Current":
            s_relative=np.array(s)
            prom=1.5
        else:
            s=np.array(s)
            mean_s=abs(np.mean(s))
            if mean_s>1:
                s_relative=s/mean_s
            else:
                s_relative=s
            prom=0.2
        #Peak and area
        window_len=51
        window='hanning'
        s_smoothing=np.r_[s_relative[window_len-1:0:-1],s_relative,s_relative[-2:-window_len-1:-1]]
        w=eval('np.'+window+'(window_len)')
        smoothed_signal=np.convolve(w/w.sum(),s_smoothing,mode='valid')[int(((window_len-1)/2)):-int(((window_len-1)/2))]
        peaks, properties = signal.find_peaks(smoothed_signal, prominence=prom, width=(5,150), distance=50)
        real_peaks, real_properties=signal.find_peaks(s_relative, prominence=prom, width=(5,150), distance=50) #NEW
        real_peak=min(real_peaks, key=lambda o:abs(o-peaks[0])) #NEw
        peaks=list(peaks)
        area_number = simps(s_area, x)

        #T1/2 Algorithm
        decay=s_area[peaks[0]:-1]
        decay_x=x[peaks[0]:-1]
        decay= np.asarray(decay)
        idx = (np.abs(decay)).argmin()
        decay=decay[:idx]
        decay_x=decay_x[:idx]

        window_len=31
        window='hanning'
        decay_s=np.r_[decay[window_len-1:0:-1],decay,decay[-2:-window_len-1:-1]]
        w=eval('np.'+window+'(window_len)')
        smoothed=np.convolve(w/w.sum(),decay_s,mode='valid')[int(((window_len-1)/2)):-int(((window_len-1)/2))]
        if s2 == "negative":
            mins=signal.argrelmax(smoothed, order=10, mode='wrap')
            if len(mins[0]) == 0:
                pass
            else:
                decay=decay[:mins[0][0]]
                decay_x=decay_x[:mins[0][0]]
            params, pcov=scipy.optimize.curve_fit(lambda t,Co,k: Co*np.exp(k*t),  decay_x,  -decay, p0=(40,-0.5))
            x_fit=decay_x
            y_fit=params[0]*np.exp(params[1]*x_fit)
            perr = np.sqrt(np.diag(pcov))
            r2_s=r2_score(-decay,y_fit) #Data needs to be linearized to calculate R-square score.
            SEOE=np.sqrt((np.sum(np.square(-decay-y_fit)))/(len(decay)))
            #p-value for Co
            tt = (params[0]-0)/np.sqrt(perr[0]/float(len(x_fit)))  # t-statistic for mean
            pval = scipy.stats.t.sf(np.abs(tt), float(len(x_fit))-1)*2  # two-sided pvalue = Prob(abs(t)>tt)


            tt2 = (params[1]-0)/np.sqrt(perr[1]/float(len(x_fit)))  # t-statistic for mean
            pval2 = scipy.stats.t.sf(np.abs(tt2), float(len(x_fit))-1)*2  # two-sided pvalue = Prob(abs(t)>tt)

            K_val=np.abs(params[1])
            T_half=(np.log(2)/K_val)
            #Experimental
            std_thalf=np.sqrt(((-(np.log(2)/(K_val**2)))**2)*(perr[1]**2))
        else:
            mins=signal.argrelmin(smoothed, order=10, mode='wrap')
            if len(mins[0]) == 0:
                pass
            else:
                decay=decay[:mins[0][0]]
                decay_x=decay_x[:mins[0][0]]
            params, pcov=scipy.optimize.curve_fit(lambda t,Co,k: Co*np.exp(k*t),  decay_x,  decay, p0=(40,-0.5))
            x_fit=decay_x
            y_fit=params[0]*np.exp(params[1]*x_fit)
            perr = np.sqrt(np.diag(pcov))
            r2_s=r2_score(decay,y_fit)
            SEOE=np.sqrt((np.sum(np.square(decay-y_fit)))/(len(decay)))
            #p-value for Co
            tt = (params[0]-0)/np.sqrt(perr[0]/float(len(x_fit)))  # t-statistic for mean
            pval = scipy.stats.t.sf(np.abs(tt), float(len(x_fit))-1)*2  # two-sided pvalue = Prob(abs(t)>tt)
            tt2 = (params[1]-0)/np.sqrt(perr[1]/float(len(x_fit)))  # t-statistic for mean
            pval2 = scipy.stats.t.sf(np.abs(tt2), float(len(x_fit))-1)*2  # two-sided pvalue = Prob(abs(t)>tt
            K_val=np.abs(params[1])
            T_half=(np.log(2)/K_val)
            std_thalf=np.sqrt(((-(np.log(2)/(K_val**2)))**2)*(perr[1]**2))

        #Sending the response
        if peaks and area_number:
            peaks_string=" ".join(str(x) for x in peaks)
            area_string=str(area_number)
            param1_string=str(params[0])
            param2_string=str(params[1])
            r2_string=str(r2_s)
            pval_string=str(pval)
            pval2_string=str(pval2)
            Th_string=str(T_half)
            SEOE=str(SEOE)
            std1_string=str(perr[0])
            std2_string=str(perr[1])
            std_thalf_string=str(std_thalf)
            decay_x_string=" ".join(str(x) for x in decay_x)
            return str(real_peak)+"&"+area_string+"&"+param1_string+"&"+param2_string+"&"+r2_string+"&"+pval_string+"&"+pval2_string+"&"+Th_string+"&"+decay_x_string+"&"+std1_string+"&"+std2_string+"&"+std_thalf_string+"&"+SEOE
        else:
            return "NaN"


    else:
        return "Error."


@app.route('/gradient', methods=['GET'])
def gradient():
    if 'signal' in request.args and 'col1' in request.args and 'sign' in request.args and 'peak' in request.args:
        x=str(request.args['signal']).split(" ")
        time_x=str(request.args['col1']).split(" ")
        sign=str(request.args['sign']);
        peak=int(request.args['peak'])
        x=[float(i) for i in x]
        time_x= [float(i) for i in time_x]
        x=np.array(x)
        time_x=np.array(time_x)
        grad=np.gradient(x,time_x)
        grad_string=" ".join(str(x) for x in grad)
        #Michaelis Menten ODE fiting
        freq=float((1/(time_x[1]-time_x[0])))
        #Calculate max point of the gradient
        if sign=="negative":
            max_grad=float(np.min(grad))
            max_grad_index=int(np.argmin(grad))
        else:
            max_grad=float(np.max(grad))
            max_grad_index=int(np.argmax(grad))
        time_st=5
        sample_st=int(time_st*freq)
        R_array=np.zeros(5*len(x))
        ramp=np.linspace(0,x[peak],max_grad_index-sample_st)
        ramp_down=np.linspace(x[peak],0,max_grad_index-sample_st)
        R_array[sample_st:max_grad_index]=ramp
        R_array[max_grad_index:max_grad_index+max_grad_index-sample_st]=ramp_down #Definition of the rate of release triangle.
        A_array=np.zeros(5*len(x))
        autoreg=np.linspace(0,0.1, len(A_array)-max_grad_index)
        A_array[max_grad_index:len(A_array)]=autoreg
        #Define the differential equation
        def f(xs, t, ps):
            R=0
            A=0
            index=int(t*freq)
            try:
                R=float(R_array[index])
            except:
                R=0
            A=float(A_array[index])
            try:
                alpha = ps['alpha'].value
                Vmax = ps['Vmax'].value
                Km=ps['Km'].value
                alpha2 = ps['alpha2'].value
                Vmax2 = ps['Vmax2'].value
                Km2=ps['Km2'].value


            except:
                alpha, Vmax,Km, alpha2, Vmax2,Km2 = ps
            S=xs
            dsdt=R*(1-A)-float(alpha)*(float(Vmax)*S/(float(Km)+S))-float(alpha2)*(float(Vmax2)*S/(float(Km2)+S))
            return dsdt
        def g(t, x0, ps):
            solution = odeint(f, x0, t, args=(ps,))
            return solution
        def residual(ps, ts, data):
            x0 = ps['x0'].value
            model = g(ts, x0, ps)
            return (model - data).ravel()
        t=time_x
        data=np.array([np.array([i]) for i in x])
        parameters = Parameters()
        parameters.add('x0', value=float(data[0]), min=0, max=100)
        parameters.add('alpha', value= 1.0, min=0, max=1)
        parameters.add('Vmax', value= 20, min=0, max=100)
        parameters.add('Km', value= 5, min=0, max=100)
        parameters.add('alpha2', value= 0, min=0, max=1)
        parameters.add('Vmax2', value= 500, min=0, max=2000)
        parameters.add('Km2', value= 150, min=0, max=1000)

# fit model and find predicted values
        result = minimize(residual, parameters, args=(t, data), method='leastsq')
        final = data + result.residual.reshape(data.shape)
        final=np.concatenate( final, axis=0 )
        final_string=" ".join(str(w) for w in final)
        alpha=str(result.params['alpha'].value)
        alpha_u=str(result.params['alpha'].stderr)
        Vmax=str(result.params['Vmax'].value)
        Vmax_u=str(result.params['Vmax'].stderr)
        Km=str(result.params['Km'].value)
        Km_u=str(result.params['Km'].stderr)
        alpha2=str(result.params['alpha2'].value)
        alpha2_u=str(result.params['alpha2'].stderr)
        Vmax2=str(result.params['Vmax2'].value)
        Vmax2_u=str(result.params['Vmax2'].stderr)
        Km2=str(result.params['Km2'].value)
        Km2_u=str(result.params['Km2'].stderr)
        #Returning the required parameters
        return grad_string+"&"+final_string+"&"+alpha+"&"+alpha_u+"&"+Vmax+"&"+Vmax_u+"&"+Km+"&"+Km_u+"&"+alpha2+"&"+alpha2_u+"&"+Vmax2+"&"+Vmax2_u+"&"+Km2+"&"+Km2_u


@app.route('/cv', methods=['GET','POST'])
def cv():
    if request.is_json:
        json_data=request.get_json()
        f = float(json_data["frequency"])
        i = str(json_data["current"]).split(" ")
        V = str(json_data["voltage"]).split(" ")
        i= np.array([float(x) for x in i])
        V= np.array([float(x) for x in V])
        time=np.linspace(0,len(V)*1000*(1/f),len(V))
        #Max voltage value to cut oxidation from reduction
        index_max = np.argmax(V)

        #Peaks in the voltage program
        V_max_points=signal.argrelmax(V, mode='wrap')[0]
        V_min_points=signal.argrelmin(V,  mode='wrap')[0]
        V_peaks=np.concatenate((V_max_points,V_min_points,np.array([len(V)])))
        #Max and min points for the current trace.
        max_points=signal.argrelmax(i, mode='wrap')[0]
        min_points=signal.argrelmin(i,  mode='wrap')[0]
        i_peaks=np.concatenate((max_points,min_points))
        #Check if peaks are faradaic or charging current.
        #Find number of samples for 400 mV
        scan_rate=int(abs((V[1]-V[0])*f))
        Vdead=0.4
        samples_dead=int((Vdead/scan_rate)*f)
        V_peaks_intervals=[[h-20, h+samples_dead] for h in V_peaks]
        V_peaks_intervals=np.sort(V_peaks_intervals, axis=0)
        dead_points=[]
        for j in range(0,len(V_peaks_intervals),1):
            for k in i_peaks:
                if k>V_peaks_intervals[j][0] and k<V_peaks_intervals[j][1]:
                    dead_points.append(k)
        dead_points=np.array(dead_points)
        i_peaksfaradaic=i_peaks[~np.isin(i_peaks,dead_points)]
        #Convert to strings
        i_peaks_string=" ".join(str(x) for x in i_peaksfaradaic)
        return i_peaks_string



    else:
        return "frequency is not there"
@app.route('/cvClass', methods=['GET','POST'])

def cvClass():
    if request.is_json:
        json_data=request.get_json()
        Ipa = float(json_data["Ipa"])
        Epa = float(json_data["Epa"])
        Epc = float(json_data["Epc"])
        AUC = float(json_data["AUC"])
        sample = pd.DataFrame({
        "Ipa":[Ipa], "Epa":[Epa], "Epc":[Epc], "AUC":[AUC]
        })

        columns=sample.columns
        pca = pk.load(open("pca.pkl",'rb'))
        scaler = pk.load(open("scaler.pkl",'rb'))
        sample = pd.DataFrame(scaler.transform(sample))
        sample.columns=columns
        result = pca.transform(sample)
        return " ".join(str(x) for x in result[0])
    else:
        return "Fail"
