import numpy as np
ndim = 4
params = ['redshift', 'nebular:fwhm_broad', 'galaxy:massformed', 'nebular:f_Ha_broad']
results = {
    'conf_int': np.array([[7.03235151, 2015.152251, 10.5212412, 7.14124e-19],[7.03924124, 2301.241241, 10.9124124, 1.12401e-18]]),
    'median': [7.03535351, 2104.1412, 10.7248124, 9.141241e-19],
}

nposterior = 100000
samples2d = np.array([np.random.normal(loc=7.0345, scale=0.001, size=nposterior), 
                      np.random.normal(loc=2100, scale=50, size=nposterior),
                      np.random.normal(loc=10.71, scale=0.1, size=nposterior),
                      np.power(10., np.random.normal(loc=-18, scale=0.2, size=nposterior))])


def _print_results():
    """ Print the 16th, 50th, 84th percentiles of the posterior. """
    # ╔═════════════════════════╦══════════╦══════════╦══════════╦════════════════════════════════════════════════════╗
    # ║ Parameter               ║   16th   ║   50th   ║   84th   ║                    Distribution                    ║
    # ╠═════════════════════════╬══════════╬══════════╬══════════╬════════════════════════════════════════════════════╣
    # ║                         ║          ║          ║          ║ 1e-20         ▁▁▁▁▂▂▂▂▂▃▃▃▄▄▅▅▆▇█▇▅▂         1e-18 ║
    
    parameter_len = 25
    print('╔' + '═'*parameter_len + '╦' + '═'*12 + '╦' + '═'*12 + '╦' + '═'*12 + '╦' + '═'*54 + '╗')
    print('║ ' + 'Parameter' + ' '*(parameter_len-10) + '║    16th    ║    50th    ║    84th    ║' + ' '*21 + 'Distribution' + ' '*21 + '║')
    print('╠' + '═'*parameter_len + '╬' + '═'*12 + '╬' + '═'*12 + '╬' + '═'*12 + '╬' + '═'*54 + '╣')
    # self.logger.info(f"{'Parameter':<25} {'16th':>10} {'50th':>10} {'84th':>10}")
    # self.logger.info("-"*58)
    for i in range(ndim):
        s = "║ "
        s += f"{params[i]}" + ' '*(parameter_len-len(params[i])-2) 
        s += " ║ "
        
        samples = samples2d[i]
        p00 = np.percentile(samples, 0.01)
        p99 = np.percentile(samples, 99.99)
        p16 = results['conf_int'][0,i]
        p50 = results['median'][i]
        p84 = results['conf_int'][1,i]

        sig_digit = int(np.floor(np.log10(np.min([p84-p50,p50-p16]))))-1
        if sig_digit >= 0: 
            p00 = int(np.round(p00, -sig_digit))
            p16 = int(np.round(p16, -sig_digit))
            p50 = int(np.round(p50, -sig_digit))
            p84 = int(np.round(p84, -sig_digit))
            p99 = int(np.round(p99, -sig_digit))
            s += f"{p16:<10d} ║ {p50:<10d} ║ {p84:<10d}"
        else:
            p00 = np.round(p00, -sig_digit)
            p16 = np.round(p16, -sig_digit)
            p50 = np.round(p50, -sig_digit)
            p84 = np.round(p84, -sig_digit)
            p99 = np.round(p99, -sig_digit)
            s += f"{p16:<10} ║ {p50:<10} ║ {p84:<10}"

        s += ' ║ '

        bins = np.linspace(np.percentile(samples, 0.01), np.percentile(samples, 99.99), 39)
        ys, _ = np.histogram(samples, bins=bins)
        ys = ys/np.max(ys)
        #  ▁▁▁▁▂▂▂▂▂▃▃▃▄▄▅▅▆▇█▇▅▂ 
        s += f"{p00:<7}"
        for y in ys:
            if y<1/16: 
                s += ' '
            elif y<3/16: # between 1/16 to 3/16 -> 1/8th height
                s += '▁'
            elif y<5/16: # between 3/16 to 5/16 -> 2/8th = 1/4th height
                s += '▂'
            elif y<7/16: # between 5/16 to 7/16 -> 3/8th height
                s += '▃'
            elif y<9/16: 
                s += '▄'
            elif y<11/16: 
                s += '▅'
            elif y<13/16: 
                s += '▆'
            elif y<15/16: 
                s += '▇'
            else:
                s += '█'
        s += f"{p99:>7} "
                
        s += '║'
    
        print(s)
    print('╚' + '═'*parameter_len + '╩' + '═'*12 + '╩' + '═'*12 + '╩' + '═'*12 + '╩' + '═'*54 + '╝')

_print_results()