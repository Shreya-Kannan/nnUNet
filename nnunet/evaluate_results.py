from nnunet.evaluation.evaluator import evaluate_folder


FOLDER_WITH_GT = '/home/shreya/scratch/Regional/Dataset_Nifti/Task502_Mid/labelsTs'
FOLDER_WITH_PREDS = '/home/shreya/scratch/Regional/nnUnet_Predictions/Registered_mid'
LABELS = (0,1,2,3,4,5,6)
#LABELS = (0,1,2,3,4)
evaluate_folder(
    FOLDER_WITH_GT,
    FOLDER_WITH_PREDS,
    LABELS, advanced=True
)

"""modes = ['Tr','Ts']
slices1 = ['Task503_Apical','Task502_Mid']
exp = ['overlayed1','corrected']

for sl in slices:
    for mode in modes:"""