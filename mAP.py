import operator

# Create a list of tuples for each class containing (filename, precision, recall) for each image
tuples_list = [(1, 0.2, 0.8), (2, 0.4, 0.6), (3, 0.6, 0.4), (4, 0.8, 0.2)]

# Order tuples (image_path, precision, recall) by increasing recall
def order_tuples(tuples_list):
  tuples_list.sort(key=lambda tup: tup[2])
  
# Find tuple with highest precision value
def max_prec(tuples_list):
  max_tuple = (0,0,0)
  for t in tuples_list:
    if t[1] > max_tuple[1]:
      max_tuple = t
  return max_tuple
  
# Calculate list of tuples of point coordinates to draw staircase precision-recall curve
points = []
def stair_coord(tuples_list, points):
  max = max_prec(tuples_list)
  points.append((max[2],max[1]))
  if len(tuples_list) > 1:
    return stair_coord(tuples_list[:tuples_list.index(max)], points)
  else:
    return points
    
# Fill in the concave points
def fill_in(points):
  coord_list = [(0, points[0][1])]
  for p in range(len(points)-1):
    coord_list.append(points[p])
    coord_list.append((points[p][0], points[p+1][1]))
  coord_list.append(points[-1])
  coord_list.append((1,points[-1][1]))
  return coord_list
  
print(stair_coord(tuples_list, points))
print(fill_in(points))