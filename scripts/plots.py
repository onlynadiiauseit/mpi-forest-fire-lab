"""Reproduce every project figure from CSV measurements and the saved GIF."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from analysis import ROOT, analyze, write_outputs


def main():
    data=analyze()
    write_outputs(data)
    folder=ROOT/'figures'
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.22,'savefig.dpi':190})

    def save(name,draw,height=2.7):
        fig,ax=plt.subplots(figsize=(6.6,height))
        draw(ax)
        fig.tight_layout(pad=.7)
        fig.savefig(folder/f'{name}.png',bbox_inches='tight')
        plt.close(fig)

    strong=data['strong']
    ps=[r['p'] for r in strong]
    for name,efficiency in [('strong_speed',False),('strong_efficiency',True)]:
        def draw(ax):
            values=[r['speed']/r['p'] if efficiency else r['speed'] for r in strong]
            lo=[values[i]-data['v0_time']/r['max']/(r['p'] if efficiency else 1) for i,r in enumerate(strong)]
            hi=[data['v0_time']/r['min']/(r['p'] if efficiency else 1)-values[i] for i,r in enumerate(strong)]
            ax.errorbar(ps,values,yerr=[lo,hi],fmt='o-',capsize=3,label='Вимірювання')
            ax.plot(ps,[1]*len(ps) if efficiency else ps,'--',color='.5',label='Ідеальне')
            ax.set(xlabel='Кількість процесів P',ylabel='Ефективність E(P)' if efficiency else 'Прискорення S(P)',xticks=ps)
            ax.legend(fontsize=8)
        save(name,draw)

    def weak(ax):
        rows=data['weak']
        ax.plot([r['p'] for r in rows],[r['time'] for r in rows],'o-',label='Час моделювання')
        ax.axhline(rows[0]['time'],ls='--',color='.5',label='Час при P=1')
        ax.set(xlabel='Кількість процесів P',ylabel='Сума максимумів кроків, с',xticks=ps)
        ax.legend(fontsize=8)
    save('weak_scaling',weak)

    def communication(ax):
        for component,label,marker in [('exchange','Обмін','o'),('reduction','Allreduce','s'),('count','Облік вогню','^')]:
            ax.plot(ps,[100*r['parts'][component]/r['time'] for r in strong],marker+'-',label=label)
        ax.set(xlabel='Кількість процесів P',ylabel='Частка загального часу, %',xticks=ps)
        ax.legend(fontsize=8)
    save('communication_share',communication)

    def profile(ax):
        keys=['compute','count','exchange','reduction','other']
        labels=['Оновлення','Облік','Обмін','Allreduce','Інше']
        bottom=[0.]*len(ps)
        for key,label in zip(keys,labels):
            values=[100*r['parts'][key]/r['time'] for r in strong]
            ax.bar([str(p) for p in ps],values,bottom=bottom,label=label)
            bottom=[a+b for a,b in zip(bottom,values)]
        ax.set(xlabel='Кількість процесів P',ylabel='Узгоджений профіль, %',ylim=(0,100))
        ax.legend(fontsize=7,ncol=3)
    save('time_profile',profile)

    def channel(ax):
        rows=data['channel']
        ax.loglog([r['bytes'] for r in rows],[r['time']*1e6 for r in rows],'o-',ms=3,label='Зустрічний обмін')
        ax.loglog([r['bytes'] for r in rows],[r['model']*1e6 for r in rows],'--',label='Модель Хокні')
        ax.set(xlabel='Розмір повідомлення, байтів',ylabel='Час, мкс')
        ax.legend(fontsize=8)
    save('channel_model',channel,2.4)

    def extra(ax):
        rows=data['extra']
        for key,label in [('ordered','Sendrecv'),('nonblocking','Irecv/Isend'),('overlap','Перекриття'),('2d','2D')]:
            ax.plot([r['p'] for r in rows],[r[key] for r in rows],'o-',label=label)
        ax.set(xlabel='Кількість процесів P',ylabel='Час, с',xticks=[r['p'] for r in rows])
        ax.legend(fontsize=8,ncol=2)
    save('additional_comparison',extra)

    def hybrid(ax):
        rows=data['hybrid']
        ax.plot([r['cores'] for r in rows],[r['mpi'] for r in rows],'o-',label='Лише MPI')
        ax.plot([r['cores'] for r in rows],[r['hybrid'] for r in rows],'s-',label='MPI+OpenMP')
        ax.set(xlabel='Однакова кількість фізичних ядер',ylabel='Час, с',xticks=[r['cores'] for r in rows])
        ax.legend(fontsize=8)
    save('hybrid_comparison',hybrid)

    def percolation(ax):
        rows=data['percolation']
        ax.errorbar([r['d'] for r in rows],[r['probability'] for r in rows],
                    yerr=[[r['probability']-r['low'] for r in rows],[r['high']-r['probability'] for r in rows]],
                    fmt='o-',ms=3,capsize=2,label='50 реалізацій, інтервал Вілсона 95%')
        ax.axvline(.5927,color='.4',ls='--',label='Теоретичний поріг 0,5927')
        ax.set(xlabel='Щільність лісу d',ylabel='Частка досягнення правого краю',ylim=(-.03,1.05))
        ax.legend(fontsize=7)
    save('percolation',percolation)

    with Image.open(folder/'поширення.gif') as gif:
        montage=Image.new('RGB',(960,320),'white')
        for index,frame in enumerate([0,gif.n_frames//2,gif.n_frames-1]):
            gif.seek(frame)
            image=gif.convert('RGB')
            image.thumbnail((310,310))
            montage.paste(image,(index*320,0))
        montage.save(folder/'animation_frames.png')
    print('Reproduced 10 figures; numerical tables are in measurements/analysis_*.csv.')


if __name__=='__main__':
    main()
