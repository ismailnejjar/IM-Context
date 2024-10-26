from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np 

def pdf_relevance(y, bandwidth=1):
    y = np.squeeze(np.asarray(y))
    y = y.reshape(len(y),1)
    pdf = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    pdf.fit(y)
    pdf_vals = np.exp(pdf.score_samples(y))
    y_relevance = 1 - (pdf_vals - pdf_vals.min())/(pdf_vals.max() - pdf_vals.min())
    
    return y_relevance

def split_domains(X, y, relevance, relevance_threshold):
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    assert len(X) == len(y) == len(relevance), ('X, y, and relevance must have the same '
                                                  'length')
    rare_indices = np.where(relevance >= relevance_threshold)[0]
    norm_indices = np.where(relevance < relevance_threshold)[0]
    assert len(rare_indices) < len(norm_indices), ('Rare domain must be smaller than '
              'normal domain. Adjust your relevance values or relevance threshold so '
              'that the there are fewer samples in the rare domain.')
    X_rare, y_rare = X[rare_indices,:], y[rare_indices]
    X_norm, y_norm = X[norm_indices,:], y[norm_indices]
    
    return [X_norm, y_norm, X_rare, y_rare]

def undersample(X, y, size, random_state=None):
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    assert len(X)==len(y), 'X and y must be of the same length.'
    if size >= len(y):
        raise ValueError('size must be smaller than the length of y')
    np.random.seed(random_state)
    new_indices = np.random.choice(range(len(y)), size, replace=False)
    X_new, y_new = X[new_indices, :], y[new_indices]
    return [X_new, y_new]  

def oversample(X, y, size, method, k=None, delta=None, relevance=None, nominal=None,
               random_state=None):
    
    # Prepare data 
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    assert len(X)==len(y), 'X and y must be of the same length.'
    moresize = int(size - len(y))
    if moresize <=0:
        raise ValueError('size must be larger than the length of y')
    
    
    # Generate extra samples for oversampling
    np.random.seed(seed=random_state)
    if method=='smoter':
        if k is None:
            raise ValueError("Must specify k if method is 'smoter'")
        [X_more, y_more] = smoter_interpolate(X, y, k, size=moresize, nominal=nominal, 
                                              random_state=random_state)
    elif method=='gaussian':
        if delta is None:
            raise ValueError("Must specify delta if method is 'gaussian'")
        [X_more, y_more] = add_gaussian(X, y, delta, size=moresize, nominal=nominal,
                                        random_state=random_state)
    else:
        raise ValueError('Wrong method specified.')
    
    # Combine old dataset with extrasamples
    X_new = np.append(X, X_more, axis=0)
    y_new = np.append(y, y_more, axis=0)
    
    return [X_new, y_new]

def smoter_interpolate(X, y, k, size, nominal=None, random_state=None):
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    assert len(X)==len(y), 'X and y must be of the same length.'
    neighbor_indices = get_neighbors(X, k)  # Get indices of k nearest neighbors
    np.random.seed(seed=random_state)
    sample_indices = np.random.choice(range(len(y)), size, replace=True) 
    X_new, y_new = [], []
        
    for i in sample_indices:
        # Get case and nearest neighbor
        X_case, y_case = X[i,:], y[i]
        neighbor = np.random.choice(neighbor_indices[i,:])
        X_neighbor, y_neighbor = X[neighbor, :], y[neighbor]
        
        # Generate synthetic case by interpolation
        rand = np.random.rand() * np.ones_like(X_case)
        
        if nominal is not None:
            rand = [np.random.choice([0,1]) if x in nominal else rand[x] \
                    for x in range(len(rand))] # Random selection for nominal features, rather than interpolation
            rand = np.asarray(rand)
        diff = (X_case - X_neighbor) * rand
        X_new_case = X_neighbor + diff
        d1 = np.linalg.norm(X_new_case - X_case)
        d2 = np.linalg.norm(X_new_case - X_neighbor)
        y_new_case = (d2 * y_case + d1 * y_neighbor) / (d2 + d1 + 1e-10) # Add 1e-10  to avoid division by zero
        X_new.append(X_new_case)
        y_new.append(y_new_case)
    
    X_new = np.array(X_new)
    y_new = np.array(y_new)
    
    return [X_new, y_new]

def get_neighbors(X, k):
    """Return indices of k nearest neighbors for each case in X"""
    
    X = np.asarray(X)
    dist = pdist(X)
    dist_mat = squareform(dist)
    order = [np.argsort(row) for row in dist_mat]
    neighbor_indices = np.array([row[1:k+1] for row in order])
    return neighbor_indices

def random_undersample(X, y, relevance_threshold=0.5, under='balance', random_state=0):
    
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = pdf_relevance(y)
    relevance = np.squeeze(np.asarray(relevance))
    [X_norm, y_norm, X_rare, y_rare] = split_domains(X, y, relevance, relevance_threshold)

    # Determine size of normal domain after undersampling
    if type(under)==float:
        assert 0 < under < 1, "under must be between 0 and 1"
        new_norm_size = int((1 - under) * len(y_norm))
    elif under=='balance':
       new_norm_size = int(len(y_rare))
    elif under=='extreme':
         new_norm_size = int(len(y_rare)**2 / len(y_norm))
         if new_norm_size <= 1:
             raise ValueError("under='extreme' results in a normal domain with {0} "
                              "samples".format(new_norm_size))
    elif under=='testing':
       new_norm_size = int(len(y_rare)/2)
    elif under=='average':
        new_norm_size = int((len(y_rare) + len(y_rare)**2 / len(y_norm)) / 2)
    else:
        raise ValueError("Incorrect value of 'under' specified.")
   
    # Undersample normal domain
    [X_norm_new, y_norm_new] = undersample(X_norm, y_norm, size=new_norm_size, random_state=random_state)
    X_new = np.append(X_norm_new, X_rare, axis=0)
    y_new = np.append(y_norm_new, y_rare, axis=0)
    
    return X_new, y_new

def smoter(X, y, relevance_threshold=0.5, k=5, over='balance', under=None, 
		   nominal=None, random_state=0):
    # Split data into rare and normal dormains
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = pdf_relevance(y)
    relevance = np.squeeze(np.asarray(relevance))
    [X_norm, y_norm, X_rare, y_rare] = split_domains(X, y, relevance, relevance_threshold)
    norm_size, rare_size = len(y_norm), len(y_rare)
    
    # Determine new sizes for rare and normal domains after oversampling
    if type(over)==float:
        assert type(under)==float, 'under must also be a float if over is a float'
        assert 0 <= under <= 1, 'under must be between 0 and 1'
        assert over >=0 , "over must be non-negative"
        new_rare_size = int((1 + over) * rare_size)
        new_norm_size = int((1 - under) * norm_size)
    elif over=='balance':
        new_rare_size = new_norm_size = int((norm_size + rare_size)/2)
    elif over == 'extreme':
        new_rare_size, new_norm_size = norm_size, rare_size
    elif over == 'average':
        new_rare_size = int(((norm_size + rare_size)/2 + norm_size)/2)
        new_norm_size = int(((norm_size + rare_size)/2 + rare_size)/2)
    else:
        raise ValueError("Incorrect value of over, must be a float or  "
                         "'balance', 'extreme', or 'average'")
        
    # Oversample rare domain
    y_median = np.median(y)
    low_indices = np.where(y_rare < y_median)[0]
    high_indices = np.where(y_rare >= y_median)[0]
    
    # First oversample low rare cases
    if len(low_indices) != 0:
        size = int(len(low_indices)/rare_size * new_rare_size)
        X_low_rare, y_low_rare = oversample(X_rare[low_indices,:], y_rare[low_indices], 
                                     size=size, method='smoter', k=k, relevance=relevance,
                                     nominal=nominal, random_state=random_state)
        
    # Then do high rare cases
    if len(high_indices) != 0:
        size = int(len(high_indices)/rare_size * new_rare_size)
        X_high_rare, y_high_rare = oversample(X_rare[high_indices], y_rare[high_indices],
                                     size=size, method='smoter', k=k, relevance=relevance,
                                     nominal=nominal, random_state=random_state)
    
    # Combine oversampled low and high rare cases
    if min(len(low_indices), len(high_indices)) != 0:
        X_rare_new = np.append(X_low_rare, X_high_rare, axis=0)
        y_rare_new = np.append(y_low_rare, y_high_rare, axis=0)
    elif len(low_indices) == 0:
        X_rare_new =  X_high_rare
        y_rare_new =  y_high_rare
    elif len(high_indices) == 0:
        X_rare_new =  X_low_rare
        y_rare_new = y_low_rare
        
    # Undersample normal cases
    X_norm_new, y_norm_new = undersample(X_norm, y_norm, size=new_norm_size, 
                                         random_state=random_state)
    
    # Combine resampled rare and normal cases
    X_new = np.append(X_rare_new, X_norm_new, axis=0)
    y_new = np.append(y_rare_new, y_norm_new, axis=0)
    
    return X_new, y_new

def inver_distribution_sampling(gaussian_labels,target_samples=1000,n_bins = 50,random_seed = 0):
    np.random.seed(random_seed)
    bins = np.linspace(np.min(gaussian_labels), np.max(gaussian_labels), n_bins+1)
    # Assign labels to bins
    indices = np.digitize(gaussian_labels, bins)
    # Calculate the number of samples in each bin
    bin_counts = np.bincount(indices)[1:]  # Skip zero as np.digitize can output bins from 1 to n_bins+1
    # Calculate inverse weights (avoid division by zero)
    inverse_weights = (1 / np.maximum(bin_counts, 1))
    inverse_weights *= target_samples/ inverse_weights.sum()  # Normalize to target samples
    selected_indices = []
    for i in range(1, n_bins+1):
        in_bin = np.where(indices == i)[0]
        np.random.shuffle(in_bin)
        # Calculate number of samples to select based on inverse weight
        n_samples = int(np.round(inverse_weights[i-1])) #round
        selected_indices.extend(in_bin[:n_samples])
    
    # Verify and visualize distribution of selected samples
    selected_labels = gaussian_labels[selected_indices]
    return selected_indices