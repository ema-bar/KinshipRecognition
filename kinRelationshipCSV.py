import scipy.io
import csv
#path of the KinFace dataset ".mat" files
matFD = scipy.io.loadmat('/KinFaceW-II/meta_data/fd_pairs.mat')
matFS = scipy.io.loadmat('KinFaceW-II/meta_data/fs_pairs.mat')
matMD = scipy.io.loadmat('/KinFaceW-II/meta_data/md_pairs.mat')
matMS = scipy.io.loadmat('/KinFaceW-II/meta_data/ms_pairs.mat')

fatFD = []
fatFS = []

motMD = []
motMS = []

dauFD = []
dauMD = []

sonFS = []
sonMS = []

#reading and creating lists
for pairFD in matFD['pairs']:
    if pairFD[1] == 1:
        fatFD.append(pairFD[2][0])
        dauFD.append(pairFD[3][0])

for pairFS in matFS['pairs']:
    if pairFS[1] == 1:
        fatFS.append(pairFS[2][0])
        sonFS.append(pairFS[3][0])

for pairMD in matMD['pairs']:
    if pairMD[1] == 1:
        motMD.append(pairMD[2][0])
        dauMD.append(pairMD[3][0])

for pairMS in matMS['pairs']:
    if pairMS[1] == 1:
        motMS.append(pairMS[2][0])
        sonMS.append(pairMS[3][0])

#division of data in 60/20/20, choose a path for training/validation/testing in which to save the divisions
with open('../trainingKin.csv', mode='w', newline='') as trainingKin:
    fieldnames = ["idP", "idC", "kin"]
    test_writer = csv.DictWriter(trainingKin, fieldnames=fieldnames)
    test_writer.writeheader()
    for n in range(150):
        print(f"{fatFD[n]}, {dauFD[n]}")
        test_writer.writerow({f"{fieldnames[0]}": f"{fatFD[n]}", f"{fieldnames[1]}": f"{dauFD[n]}", f"{fieldnames[2]}": "1"})

    for n in range(150):
        print(f"{fatFS[n]}, {sonFS[n]}")
        test_writer.writerow({f"{fieldnames[0]}": f"{fatFS[n]}", f"{fieldnames[1]}": f"{sonFS[n]}", f"{fieldnames[2]}": "2"})

    for n in range(150):
        print(f"{motMD[n]}, {dauMD[n]}")
        test_writer.writerow({f"{fieldnames[0]}": f"{motMD[n]}", f"{fieldnames[1]}": f"{dauMD[n]}", f"{fieldnames[2]}": "3"})

    for n in range(150):
        print(f"{motMS[n]}, {sonMS[n]}")
        test_writer.writerow({f"{fieldnames[0]}": f"{motMS[n]}", f"{fieldnames[1]}": f"{sonMS[n]}", f"{fieldnames[2]}": "4"})

with open('../validationKin.csv', mode='w', newline='') as validationKin:
    fieldnames = ["idP", "idC", "kin"]
    test_writer = csv.DictWriter(validationKin, fieldnames=fieldnames)
    test_writer.writeheader()
    for n in range(150, 200):
        print(f"{fatFD[n]}, {dauFD[n]}")
        test_writer.writerow({f"{fieldnames[0]}": f"{fatFD[n]}", f"{fieldnames[1]}": f"{dauFD[n]}", f"{fieldnames[2]}": "1"})

    for n in range(150, 200):
        print(f"{fatFS[n]}, {sonFS[n]}")
        test_writer.writerow({f"{fieldnames[0]}": f"{fatFS[n]}", f"{fieldnames[1]}": f"{sonFS[n]}", f"{fieldnames[2]}": "2"})

    for n in range(150, 200):
        print(f"{motMD[n]}, {dauMD[n]}")
        test_writer.writerow({f"{fieldnames[0]}": f"{motMD[n]}", f"{fieldnames[1]}": f"{dauMD[n]}", f"{fieldnames[2]}": "3"})

    for n in range(150, 200):
        print(f"{motMS[n]}, {sonMS[n]}")
        test_writer.writerow({f"{fieldnames[0]}": f"{motMS[n]}", f"{fieldnames[1]}": f"{sonMS[n]}", f"{fieldnames[2]}": "4"})

with open('../testingKin.csv', mode='w', newline='') as testingKin:
    fieldnames = ["idP", "idC", "kin"]
    test_writer = csv.DictWriter(testingKin, fieldnames=fieldnames)
    test_writer.writeheader()
    for n in range(200, 250):
        print(f"{fatFD[n]}, {dauFD[n]}")
        test_writer.writerow({f"{fieldnames[0]}": f"{fatFD[n]}", f"{fieldnames[1]}": f"{dauFD[n]}", f"{fieldnames[2]}": "1"})

    for n in range(200, 250):
        print(f"{fatFS[n]}, {sonFS[n]}")
        test_writer.writerow({f"{fieldnames[0]}": f"{fatFS[n]}", f"{fieldnames[1]}": f"{sonFS[n]}", f"{fieldnames[2]}": "2"})

    for n in range(200, 250):
        print(f"{motMD[n]}, {dauMD[n]}")
        test_writer.writerow({f"{fieldnames[0]}": f"{motMD[n]}", f"{fieldnames[1]}": f"{dauMD[n]}", f"{fieldnames[2]}": "3"})

    for n in range(200, 250):
        print(f"{motMS[n]}, {sonMS[n]}")
        test_writer.writerow({f"{fieldnames[0]}": f"{motMS[n]}", f"{fieldnames[1]}": f"{sonMS[n]}", f"{fieldnames[2]}": "4"})


trainingKin.close()
validationKin.close()
testingKin.close()