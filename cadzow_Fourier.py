# The function cadzow implements the Cadzow Algorithm using Fourier in the measurements.



def avrg_Hankel(A) :
    """
    this function returns a Hankel matrix formed by averaging across the antidiagonals of A. This is part
    of the denoising process in the Cadzow Algorithm
    inputs: A --> 2d numpy array
    """
    dim = A.shape
    flip_A = np.fliplr(A) # flip A to get its anti diagonals c[0]= A[0,0]
    col = zeros(dim[0],dtype = complex)
    row = zeros(dim[1], dtype = complex) # col and row are the vectors of the averages to create the new Hankel matrix
    for k in range(dim[0]):
        col[k] = mean(diag(flip_A, dim[0]-1-k))
    # create r    
    for k in range(dim[1]):
        row[k] = mean(diag(flip_A,-k))
    
    H = hankel(col, row)
    #print(col,row)
    return H, col, row

def cadzow(Y_tilde, kmax, d, r):
    """
    this function denoises the meassurements matrix Ytilde using the Cadzow algorithm.
    Inputs : Y_tilde --> Jx(L+1) array: noisy meassurements for all the snapshots. 
             kmax --> int: indicating max number of iterations to denoise
    """
    # first generate the fourier transform along columns of the matrix Ytilde
    J = Y_tilde.shape[0]
    L = Y_tilde.shape[1] - 1
    m = d//J    # no me queda claro si este es m el que me da el rango del SVD
    
    # computation of normalized Y^ column by column
    Y_hat = scipy.fft.fft(Y_tilde, axis = 0, norm = 'ortho') 
    
    # keeping for each j the max and min singular value of X
    singular_val = []
    
    for j in range(J) :
        # Form Hankel matrix X = H(j)
        col = Y_hat[j,0: L//2 + 1]
        row = Y_hat[j, L//2 : L+1]
        X = hankel(col, row) 
        
        #denoising:
        for k in range(kmax):
            U, Sigma, Vh = linalg.svd(X, full_matrices = True) # non econ!
                        
            #truncating the svd to rank r
            S = Sigma[0:r]
            Ur = U[:,0:r]
            Vh_r = Vh[0:r,:]
            X = Ur @ diag(S) @ Vh_r
            # the new Hankel matrix obtained by averaging X across the anti diagonals
            Hnew, col_new, row_new = avrg_Hankel(X)
            X = Hnew
            
        # update row j of Y^: have in mind that col_new[-1] = row_new[0]
        Y_hat[j,0:L//2 + 1] =  col_new   #Hnew[:,0]
        Y_hat[j, L//2+1: L+1] =  row_new[1:]        #Hnew[-1,1:]
    
    # update Ytilde by taking the inverse fourier transform of the updated Y^
    Y_tilde = scipy.fft.ifft(Y_hat, axis = 0, norm='ortho')
    #print(singular_val)
    
    return Y_tilde