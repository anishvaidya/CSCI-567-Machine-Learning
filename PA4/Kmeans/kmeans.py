import numpy as np

def distance_from_centroids(point, x, distances):
    result = []
    for i in range(len(x)):
        if len(distances)==0:
            res=[]
        else:    
            res = [distances[i]]
        res.append(np.square(np.linalg.norm(x[i] - x[point])))
        result.append(min(res))    
    return result


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
#    raise Exception(
#             'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')
    centers = []
    centers.append(generator.randint(0, n))
    distances = []
    while len(centers) < n_cluster:
        new_distances = distance_from_centroids(centers[-1], x, distances)
        centers.append(np.argmax(new_distances/sum(new_distances)))
        distances = new_distances

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
#        raise Exception(
#             'Implement fit function in KMeans class')
        
        def compute_distortion(centroids, x, y):
            distortion = np.sum([np.sum((x[y == i] - centroids[i])) for i in range(self.n_cluster)])
            return distortion/x.shape[0]

        centroids = np.zeros((self.n_cluster, D))
        for i in range(len(self.centers)):
            centroids[i] = x[self.centers[i]]
        y = np.zeros(N)
        distortion = compute_distortion(centroids, x, y)
        iter_num = 0
        while iter_num < self.max_iter:
            y = np.argmin(np.sum(((x - np.expand_dims(centroids, axis=1))**2), axis=2), axis=0)
            nth_distortion = compute_distortion(centroids, x, y)
            if abs(distortion - nth_distortion) <= self.e:
                break
        
            distortion = nth_distortion
            z = 0
            nth_centroids = np.array([np.mean(x[y == cluster_ind], axis=0) for cluster_ind in range(self.n_cluster)])
#            for cluster_ind in range(self.n_cluster):
#                nth_centroids = np.array([np.mean(x[y == cluster_ind], axis=0)])
#            print (nth_centroids)    
            nth_centroids[np.where(np.isnan(nth_centroids))] = centroids[np.where(np.isnan(nth_centroids))]
            centroids = nth_centroids
            z = z + 2
            iter_num += 1

        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
#        raise Exception(
#             'Implement fit function in KMeansClassifier class')
        kmeans = KMeans(self.n_cluster, self.max_iter, self.e)
        centroids, assignment, n_iter = kmeans.fit(x)
        polling = list()
        for z in range(self.n_cluster):
            polling.append({})
        for point, a in zip(y, assignment):
            if point not in polling[a].keys():
                polling[a][point] = 1
            else:
                polling[a][point] += 1

        centroid_labels = list()
        for poll in polling:
            if not poll:
                centroid_labels.append(0)
            else:
                centroid_labels.append(max(poll, key=poll.get))

        centroid_labels = np.array(centroid_labels)

        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
#        raise Exception(
#             'Implement predict function in KMeansClassifier class')
        l2_norm  = np.sum(np.power((x - np.expand_dims(self.centroids,axis = 1)), 2), axis = 2)
#        print ("l2 norm", l2_norm.shape)
#        a = np.linalg.norm(x - np.expand_dims(self.centroids, axis = 1), axis = 2)
#        print ("a shape", a.shape)
#        l2_norm = np.sum(a, axis = 1)
#        print ("new l2", l2_norm.shape)
#        if ()
        calculation = np.argmin(l2_norm, axis=0)
        labels = self.centroid_labels[calculation]

        
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
#    raise Exception(
#             'Implement transform_image function')
#    
    N, M, C = image.shape
    data = image.reshape(N * M, C)
    r = np.argmin(np.sum(((data - np.expand_dims(code_vectors, axis=1)) ** 2), axis=2), axis=0)
    new_im = code_vectors[r].reshape(N, M, C)

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

