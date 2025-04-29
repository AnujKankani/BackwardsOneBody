#This is the simplest possible BOB example to demonstrate how easy it is
#See detailed_documentation.py for more info
#Or wait till I build proper documentation

import matplotlib.pyplot as plt
import BOB_utils

BOB = BOB_utils.BOB()
BOB.initialize_with_sxs_data("SXS:BBH:2325") #equal mass and non spinning case
BOB.what_should_BOB_create = "news" #options are psi4, news, strain, strain_using_psi4, strain_using_news
t,y = BOB.construct_BOB()
NR_news_t, NR_news_y = BOB.get_news_data()

plt.plot(NR_news_t,NR_news_y.real,label='SXS',color='black')
plt.plot(t,y.real,label='BOB')
plt.xlim(9400,9500)
plt.ylim(-0.2,0.2)
plt.xlabel('time')
plt.ylabel('Re(News)')
plt.title('News SXS vs BOB')
plt.legend()
plt.show()
