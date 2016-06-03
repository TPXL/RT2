from PIL import Image
from PIL import ImageFilter
from dionysus import Simplex, Filtration, StaticPersistence, vertex_cmp, data_cmp, data_dim_cmp, DynamicPersistenceChains
from sets import Set
import matplotlib.pyplot as pyplot
import matplotlib.image as mpimg
import numpy as np
import dionysus
import time
import math

def nonmaxsup(pixels, size):
    for i in range(1, size[0]-1):
        for j in range(1, size[1]-1):
            if (pixels[i, j+1] >= pixels[i, j] or
                pixels[i, j-1] >= pixels[i, j] or
                pixels[i+1, j] >= pixels[i, j] or
                pixels[i-1, j] >= pixels[i, j] or
                pixels[i+1, j+1] >= pixels[i, j] or
                pixels[i+1, j-1] >= pixels[i, j] or
                pixels[i-1, j+1] >= pixels[i, j] or
                pixels[i-1, j-1] >= pixels[i, j]):
                pixels[i, j] = 0

def derivative(pixels, size):
    for i in range(0, size[0]-1):
        for j in range(0, size[1]-1):
            dx = pixels[i, j] - pixels[i+1, j]
            dy = pixels[i, j] - pixels[i, j+1]
            pixels[i, j] = int(math.sqrt(dx**2 + dy**2))

def threshold(pixels, size, threshold = 127):
    for i in range(0, size[0]-1):
        for j in range(0, size[1]-1):
            if(pixels[i, j] < threshold):
                pixels[i, j] = 0
            else:
                pixels[i, j] = 255

def rgb2gray(processedPixels, originalPixels, size):
    for i in range(0, size[0]-1):
        for j in range(0, size[1]-1):
            processedPixels[i, j] = int((originalPixels[i, j][0] + originalPixels[i, j][1] + originalPixels[i, j][2])/3)

def plotImage(image, subplot):
    pyplot.figure(0)
    pyplot.subplot(3, 3, subplot).imshow(np.asarray(image), cmap="gray")

def pointsFromImage(pixels, size):
    retval = []
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if pixels[i, j] > 0:
                retval.append([i, j])
    return retval

def processImage(imagename, debug = False, filterRadius=0, imageThreshold=5):

    original = Image.open(imagename)
    originalPixels = original.load()
    if debug:
        plotImage(original, 1)
    
    originalSmooth = original.filter(ImageFilter.GaussianBlur(radius=filterRadius))
    originalSmoothPixels = originalSmooth.load();
    if debug:
        plotImage(originalSmooth, 2)

    processed = Image.new("P", original.size)
    processedPixels = processed.load()

    rgb2gray(processedPixels, originalSmoothPixels, original.size)
    if debug:
        plotImage(processed, 3)
    derivative(processedPixels, processed.size)
    if debug:
        plotImage(processed, 4)
    nonmaxsup(processedPixels, processed.size)
    if debug:
        plotImage(processed, 5)
    threshold(processedPixels, processed.size, threshold = imageThreshold)
    if debug:
        plotImage(processed, 6)
    
    retval = pointsFromImage(processedPixels, processed.size)
    return retval

def plot_points(points, plotnum, figure=None, radius=0, color='r'):
    """
    Plot the given list of points on the given figure using matplotlib. 
    If figure is not specified plot on the current figure.
    
    If radius is bigger than zero, then around each
    point the ball centered in the point with the 
    given radius and color will be drawn. If color is not given 
    circles are drawn using red color.
    """
    xs, ys = map(list, zip(*points))
    pyplot.axis([min(xs)-1, max(xs)+1,min(ys)-1,max(ys)+1])
    pyplot.subplot(3, 3, plotnum).invert_yaxis()
    pyplot.subplot(3, 3, plotnum).plot(xs, ys, 'ro')
    if radius > 0:
        if figure is None:
            figure = pyplot.gcf()
        axes = figure.gca()
        for circle in [pyplot.Circle(point, radius, color=color)
                       for point in points]:
            axes.add_artist(circle)

def rips(points, skeleton, max):
    """
    Generate the Vietoris-Rips complex on the given set of points in 2D.
    Only simplexes up to dimension skeleton are computed.
    The max parameter denotes the distance cut-off value.
    """
    distances = dionysus.PairwiseDistances(points)
    rips = dionysus.Rips(distances)
    simplices = dionysus.Filtration()
    rips.generate(skeleton, max, simplices.append)
    #print time.asctime(), "Generated complex: %d simplices" % len(simplices)
    for s in simplices: s.data = rips.eval(s)
    #print time.asctime(), simplices[0], '...', simplices[-1]
    return [list(simplex.vertices) for simplex in simplices]

def alpha(points, radius):
    f = dionysus.Filtration()
    dionysus.fill_alpha_complex(points, f)
    ret = [list(s.vertices) for s in f if s.data[0] < radius]
    print "Total number of simplices:", len(ret)
    return ret

def get_points(points, indices):
    '''
    Get data from point array on the given indices.
    Useful since simplex spanned by a list of points
    is given as a list of positions of the points
    in a points array.
    '''
    return [points[index] for index in indices]

def draw_triangle(triangle):
    '''
    Draw a triangle on the current figure. 
    Triangle must be given as a list of three 2D points, 
    each point as a list of two numbers.
    '''
    p1, p2, p3 = triangle
    pyplot.plot([p1[0], p2[0]],[p1[1],p2[1]])
    pyplot.plot([p1[0], p3[0]],[p1[1],p3[1]])
    pyplot.plot([p2[0], p3[0]],[p2[1],p3[1]])
        
def draw_line(line):
    '''
    Draw a line on the current figure.
    Line must be given as a list of two 2D points, 
    each point as a list of two numbers.    
    '''
    p1, p2 = line
    pyplot.plot([p1[0], p2[0]],[p1[1],p2[1]])
    
def draw_point(point):
    '''
    Draw a point on the current figure.
    Point must be given as a list of two numbers.    
    '''
    pyplot.plot(point)

def draw_simplicial_complex(simplices, points):
    '''
    Draw 2D simplicial complex on the current figure. 
    Input must be a list of simplices, each simplex a
    list of indices in the points array. 
    '''
    handlers = [draw_point, draw_line, draw_triangle]
    for simplex in simplices:
        handlers[len(simplex)-1](get_points(points, simplex))

def cv_method():
    images = ["1.tiff", "2.tiff", "3.tiff", "4.tiff", "5.tiff", "6.tiff", "7.tiff"]
    #images = ["2.tiff"]
    debug = True
    for im in images:
        print "Processing"
        points = processImage(im, debug=debug)
        print "point count: ", len(points)
        if debug:
            plot_points(points, 7, figure=pyplot.figure(0))
            pyplot.show()
            
        radius = 20000
        cx = alpha(points, radius)
        print "complex length: ", len(cx)
        draw_simplicial_complex(cx, points)
        pyplot.gca().invert_yaxis()
        pyplot.show()
        print "Processed"


def cubic_complex(pixels, size, thresh):
    retval = []
    #size = [size[0]/14, size[1]/14]
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if(pixels[i, j] > thresh):
                retval.append((i*size[1] + j,))
                if i < size[0]-1 and pixels[i+1, j] > thresh:
                    retval.append(((i+1)*size[1]+j,))
                    retval.append(((i+1)*size[1] + j, i*size[1] + j))
                if j < size[1]-1 and pixels[i, j+1] > thresh:
                    retval.append((i*size[1] + j + 1,))
                    retval.append((i*size[1] + j + 1, i*size[1] + j))
                if i < size[0]-1 and pixels[i+1, j] > thresh and j < size[1]-1 and pixels[i, j+1] > thresh and pixels[i+1, j+1] > thresh:
                    a = i*size[1] + j
                    b = (i+1)*size[1] + j
                    c = (i+1) * size[1] + j +1
                    d = i*size[1] + j + 1
                    retval.append((a, c))
                    retval.append((b, d))
                    retval.append((a, b, c))
                    retval.append((a, c, d))
                    retval.append((a, b, d))
                    retval.append((b, c, d))
                    retval.append((a, b, c, d))
    return list(retval)
        
        
def comp(lhs, rhs):
    if len(lhs) != len(rhs):
        return len(lhs)-len(rhs)
    for i in range(0, len(lhs)):
        if(lhs[i] != rhs[i]):
            return lhs[i] - rhs[i]
    return 0

        
def cubic_complex_method():
    images = ["1.tiff"]
    #images = ["1.tiff", "2.tiff", "3.tiff", "4.tiff", "5.tiff", "6.tiff", "7.tiff"]
    thresholds = range(255, -1, -1)
    #thresholds = [239]
    for im in images:
        original = Image.open(im)
        originalPixels = original.load()
        processed = Image.new("P", original.size)
        processedPixels = processed.load()
        rgb2gray(processedPixels, originalPixels, original.size)
        points = [[x, y] for x in range(0, processed.size[0]) for y in range(0, processed.size[1])]
        print "points len: ", len(points)
        print "image size: ", original.size[0] * original.size[1]
        for thresh in thresholds: 
            cc = cubic_complex(processedPixels, processed.size, thresh)
            print "cc size(", thresh, "): ", len(cc)
            #draw_simplicial_complex(cc, points)
            #pyplot.gca().invert_yaxis()
            #pyplot.show()
            cc = sorted(cc, cmp=comp)
            scx = [Simplex(cc[i], i) for i in range(0, len(cc))]
            f = Filtration(scx, data_cmp)
            p = DynamicPersistenceChains(f)
            p.pair_simplices()
            smap = p.make_simplex_map(f)
            
            print "{:>10}{:>10}{:>10}{:>10}".format("First", "Second", "Birth", "Death")
            counter = 0
            for i in (i for i in p if i.sign()):
                b = smap[i]
                if i.unpaired():
                    #print "{:>30}{:>30}{:>10}{:>10}".format(b, '', b.data, "inf")
                    counter += 1 
                #else:
                #    d = smap[i.pair()]
                #    print "{:>30}{:>30}{:>10}{:>10}".format(b, d, b.data, d.data)
            print "# inf simplices: ", counter
            
    
cubic_complex_method()

'''
scx = [Simplex((2,),        0),                 # C
       Simplex((0,),        1),                 # A
       Simplex((1,),        2),                 # B
       Simplex((0,1),       3),                 # AB
       Simplex((1,2),       4),                 # BC
       Simplex((0,2),       5),                 # AC
       Simplex((0,1,2),     6),                 # ABC
]


f = Filtration(scx, data_cmp)            
p = DynamicPersistenceChains(f)
p.pair_simplices()
smap = p.make_simplex_map(f)


print "{:>10}{:>10}{:>10}{:>10}".format("First", "Second", "Birth", "Death")
for i in (i for i in p if i.sign()):
    b = smap[i]
    if i.unpaired():
        print "{:>10}{:>10}{:>10}{:>10}".format(b, '', b.data, "inf")
    else:
        d = smap[i.pair()]
        print "{:>10}{:>10}{:>10}{:>10}".format(b, d, b.data, d.data)
'''





    



