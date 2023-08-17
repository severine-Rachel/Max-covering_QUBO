import matplotlib.pyplot as plt
liste = [4, 5, 6,7 ,1, 5, 6]
liste2 = [6, 6, 7 , 8 , 2, 4,5]
liste3 = [6, 6, 9 , 8 , 9, 4,5]
l = [liste, liste2, liste3]
fig,ax = plt.subplots()
ax.set_xticklabels(["liste1","b","c"] )
ax.set_title("Moyenne")
ax.set_ylabel("notes")
ax.set_xlabel("classes")
ax.boxplot(l, positions=[0,1,2])
plt.show()