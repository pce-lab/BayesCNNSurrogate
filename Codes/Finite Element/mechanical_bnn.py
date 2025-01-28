from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage import measure, color, io, data, filters, morphology, segmentation, util
import dolfin as dl
from dolfin import NonlinearProblem
from ufl import nabla_div
import glob
import cv2
import pandas as pd


def process_image(image_path):

    pixels_to_um = 1

    ret,thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)

    sure_bg = cv2.dilate(opening, kernel, iterations=1)

    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)

    ret2, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)


    unknown = cv2.subtract(sure_bg, sure_fg)

    ret3, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 10
    markers[unknown==255] = 0


    markers = cv2.watershed(img1,markers)

    img1[markers == -1] = [0, 255, 255]
    img2 = color.label2rgb(markers, bg_label=0)

    props = measure.regionprops_table(markers, intensity_image=img,
                                    properties = ['label',
                                                    'centroid',
                                                'area',
                                                    'equivalent_diameter',
                                                    'perimeter'])






    df = pd.DataFrame(props)

    df = df.drop(0)
    print(df)
    df.to_csv('file_name.csv')
    label_list = list(df["label"])
    dia_list1 = list(df["equivalent_diameter"])
    dia_list = [item * pixels_to_um for item in dia_list1]
    dic = dict(zip(label_list, dia_list))
    z = np.unique(label_list)
    t = np.unique(dia_list)
    print(dia_list)
    print(z)

    (nx, ny) = markers.shape

    print(markers.shape)

    #------------Characteristic length-------------

    l = 17.5e-6*4.0
    #Material parameters
    nu = Constant(0.3)
    Es= Constant(100e6)
    Ef = Constant(17.5e6)


    rho_s= 2650
    rho_f= 1.836
    c_s= 700
    c_f= 850
    rho=1

    #-------Mesh-------
    mesh = dl.RectangleMesh(dl.Point(0.,0.),dl.Point(1.*l,1.*l), nx, ny)
    #meshR_input = 72
    #domain = Polygon( [Point(0.,0.),Point(0.,1.*l),Point(1.*l,1.*l),
    #                    Point(1.*l,.5*l),Point(.5*l,.5*l),Point(.5*l,0),]  )

    V = VectorFunctionSpace(mesh, 'P', 1)
    mu = np.zeros((nx,ny))
    lambd = np.zeros((nx,ny))
    #tau = np.zeros((nx,ny))
    def pixel():
        for i in range(len(markers[:,0])):
            for j in range(len(markers[0,:])):
                for k in dic:
                    if markers[i][j] == k:
                        mu[i][j] = Ef/(2*(1+nu))
                        lambd[i][j] = (Ef*nu)/((1 + nu)*(1 - 2*nu))
                        break
                    else:
                        mu[i][j] = Es/(2*(1+nu))
                        lambd[i][j] = (Es*nu)/((1 + nu)*(1 - 2*nu))

        return [mu, lambd]

    d = pixel()
    mu1 = d[0]
    lambd1 = d[1]
    #print(E1)



    class FE_image(UserExpression):
        def eval_cell(self, value, x, ufc_cell):
            p = Cell(mesh, ufc_cell.index).midpoint()
            i, j = int(p[0]*(nx)), int(p[1]*(ny))
            value[:] = mu1[-(j+1), i]

        def value_shape(self):
            return ()

    y = FE_image()
    print(y)

    class lam(UserExpression):
        def eval_cell(self, value, x, ufc_cell):
            p = Cell(mesh, ufc_cell.index).midpoint()
            i, j = int(p[0]/l*(nx)), int(p[1]/l*(ny))
            value[:] = lambd1[-(j+1), i]

        def value_shape(self):
            return ()


    class mua(UserExpression):
        def eval_cell(self, value, x, ufc_cell):
            p = Cell(mesh, ufc_cell.index).midpoint()
            i, j = int(p[0]/l*(nx)), int(p[1]/l*(ny))
            value[:] = mu1[-(j+1), i]

        def value_shape(self):
            return ()

    mu = mua()

    lamb = lam()
    #print(lamb)
    #mu = 0.6
    lambda_ = 2.0
    def epsilon(u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)
        #return sym(nabla_grad(u))

    def sigma(u):
        return lamb*div(u)*Identity(d) + 2*mu*epsilon(u)

    class Top(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-14
            return on_boundary and (x[1]>1*l -tol)




    #f = Constant((0, -rho*g))
    #T = Constant((7071, 7071))
    tol = 1E-14

    def clamped_boundary(x, on_boundary):
        return on_boundary and x[1] < tol
    u = TrialFunction(V)
    bc = DirichletBC(V, Constant((0, 0)), clamped_boundary)
    d = u.geometric_dimension()
    v = TestFunction(V)

    # Applied force
    T = Constant((0, -1e6))

    #---------Defining Subdomains--------------
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    top= Top()

    top.mark(boundaries, 2)
    ds_v = ds(subdomain_data = boundaries)


    a = inner(sigma(u), epsilon(v))*dx
    L = dot(T, v)*ds_v(2)

    u = Function(V)
    solve(a == L, u, bc)
    dis = File("ucc2.pvd")
    dis<<u
    #plot(u, title='Displacement', mode='displacement')

    #stressb = sigma(u)
    st = File("stcc2.pvd")

    #Computing Strain energy
    eU = sym(grad(u))
    S1 = 2.0*mu*inner(eU,eU) + lamb*(tr(eU)**2)
    compliance = assemble(S1* dx)
    #S1 = assemble(S1)
    print(compliance)
    print('Strain energy for {}: {}'.format(image_path, compliance))

    with open('comp.txt','w+') as f:
        f.write(str(compliance))
    f.close()

    # Computing von mises stress 
    s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d)  # deviatoric stress
    von_Mises = sqrt(3./2*inner(s, s))
    V = FunctionSpace(mesh, 'P', 1)
    von_Mises = project(von_Mises, V)
    st<<von_Mises
    return compliance


# Loop through 100 images
strain_energies = []

# Get all .png files in the specified folder
image_files = glob.glob("BayesCNN/Aerogel_data/last500/Danial_largeFoam/bordered/*.png")

for img_path in image_files:
    try:
        img1 = cv2.imread(img_path)
        if img1 is None:
            print(f"Error: Image {img_path} not found or could not be read. Skipping...")
            continue
        
        img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Process the image and get strain energy
        strain_energy = process_image(img)
        strain_energies.append(strain_energy)

        print(f'Strain energy for {img_path}: {strain_energy}')

        # Save strain energy data for each image separately
        strain_energy_data = pd.DataFrame({'Image': [img_path], 'Strain Energy': [strain_energy]})
        # Append the strain energy data to a CSV file
        with open('BayesCNN/Aerogel_data/last500/Danial_largeFoam/bordered/strain_energies_test.csv', 'a') as f:
            strain_energy_data.to_csv(f, header=f.tell()==0, index=False)  # Append if file exists, write headers only if the file is empty

    except Exception as e:
        print(f"An error occurred while processing {img_path}: {e}")