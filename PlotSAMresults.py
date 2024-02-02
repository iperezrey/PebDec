import matplotlib.pyplot as plt
import cv2 as cv


figsize = (10, 10)
rows = 1
columns = 3

""" Plot the images with the masks """
img_num = '6673'

img_original = cv.imread('results/' + img_num + '/IMG_' + img_num + '_original (1).JPEG')
img_original = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)

img_masked = cv.imread('results/' + img_num + '/IMG_' + img_num + '_42_masked (1).JPEG')
img_masked = cv.cvtColor(img_masked, cv.COLOR_BGR2RGB)

masks = cv.imread('results/' + img_num + '/IMG_' + img_num + '_42_masks (1).JPEG')
masks = cv.cvtColor(masks, cv.COLOR_BGR2RGB)

#Create the figure
fig = plt.figure(figsize=figsize)

#Add firts image
fig.add_subplot(rows, columns, 1)

plt.imshow(img_original)
plt.axis('off')
plt.title('IMG '+ img_num)

#Add second image
fig.add_subplot(rows, columns, 2)

plt.imshow(img_masked)
plt.axis('off')
plt.title('IMG ' + img_num + ' masked')

#Add third image
fig.add_subplot(rows, columns, 3)

plt.imshow(masks)
plt.axis('off')
plt.title('IMG ' + img_num + ' masks')

plt.savefig('results/' + img_num + '/' + img_num + 'figure.JPEG', dpi=300)
plt.show()

""" Plot the histograms """
# figsizeh = (15, 15)
# rowsh = 1
# columnsh = 2

# #Histograms points per side = 42
# his_6648 = cv.imread('results/histograms/6648.jpg')
# his_6648 = cv.cvtColor(his_6648, cv.COLOR_BGR2RGB)

# his_6662 = cv.imread('results/histograms/6662.jpg')
# his_6662 = cv.cvtColor(his_6662, cv.COLOR_BGR2RGB)

# his_6701 = cv.imread('results/histograms/6701.jpg')
# his_6701 = cv.cvtColor(his_6701, cv.COLOR_BGR2RGB)

# his_6710 = cv.imread('results/histograms/6710.jpg')
# his_6710 = cv.cvtColor(his_6710, cv.COLOR_BGR2RGB)

# his_6716 = cv.imread('results/histograms/6716.jpg')
# his_6716 = cv.cvtColor(his_6716, cv.COLOR_BGR2RGB)

# his_6750 = cv.imread('results/histograms/6750.jpg')
# his_6750 = cv.cvtColor(his_6750, cv.COLOR_BGR2RGB)

# his_6762 = cv.imread('results/histograms/6762.jpg')
# his_6762 = cv.cvtColor(his_6762, cv.COLOR_BGR2RGB)

# #Histograms points per side = 18-20
# his_6666 = cv.imread('results/histograms/6666.jpg')
# his_6666 = cv.cvtColor(his_6666, cv.COLOR_BGR2RGB)

# his_6720 = cv.imread('results/histograms/6720.jpg')
# his_6720 = cv.cvtColor(his_6720, cv.COLOR_BGR2RGB)

# his_6682 = cv.imread('results/histograms/6682.jpg')
# his_6682 = cv.cvtColor(his_6682, cv.COLOR_BGR2RGB)


# #Create fig1
# fig1, axes = plt.subplots(nrows=rowsh, ncols=columnsh, figsize=(15,7))

# axes[0].imshow(his_6648, aspect='auto')
# axes[0].axis('off')
# axes[1].imshow(his_6662, aspect='auto')
# axes[1].axis('off')
# plt.subplots_adjust(wspace=0)
# plt.savefig('results/histograms/Figure1.jpg', dpi=300)
# # plt.show()


# #Create fig2
# fig2, axes = plt.subplots(nrows=rowsh, ncols=columnsh, figsize=(15,7))

# axes[0].imshow(his_6701, aspect='auto')
# axes[0].axis('off')
# axes[1].imshow(his_6710, aspect='auto')
# axes[1].axis('off')
# plt.subplots_adjust(wspace=0)
# plt.savefig('results/histograms/Figure2.jpg', dpi=300)
# # plt.show()


# #Create fig3
# fig3, axes = plt.subplots(nrows=rowsh, ncols=columnsh, figsize=(15,7))

# axes[0].imshow(his_6716, aspect='auto')
# axes[0].axis('off')
# axes[1].imshow(his_6750, aspect='auto')
# axes[1].axis('off')
# plt.subplots_adjust(wspace=0)
# plt.savefig('results/histograms/Figure3.jpg', dpi=300)
# # plt.show()


# #Create fig4
# fig4, axes = plt.subplots(nrows=rowsh, ncols=columnsh, figsize=(15,7))

# axes[0].imshow(his_6762, aspect='auto')
# axes[0].axis('off')
# # axes[1].imshow(his_6750, aspect='auto')
# axes[1].axis('off')
# plt.subplots_adjust(wspace=0)
# plt.savefig('results/histograms/Figure4.jpg', dpi=300)
# # plt.show()


# #Create fig5
# fig5, axes = plt.subplots(nrows=rowsh, ncols=columnsh, figsize=(15,7))

# axes[0].imshow(his_6662, aspect='auto')
# axes[0].axis('off')
# axes[1].imshow(his_6720, aspect='auto')
# axes[1].axis('off')
# plt.subplots_adjust(wspace=0)
# plt.savefig('results/histograms/Figure5.jpg', dpi=300)
# # plt.show()


# #Create fig6 --- IMG 6682 --- no results
# fig, axes = plt.subplots(nrows=rowsh, ncols=columnsh, figsize=(15,7))

# axes[0].imshow(his_6682, aspect='auto')
# axes[0].axis('off')
# # axes[1].imshow(his_6720, aspect='auto')
# axes[1].axis('off')
# plt.subplots_adjust(wspace=0)
# plt.savefig('results/histograms/Figure6.jpg', dpi=300)
# # plt.show()