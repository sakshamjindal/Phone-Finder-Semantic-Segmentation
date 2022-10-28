import torch

def orientation (
      point1: torch.Tensor, 
      point2: torch.Tensor, 
      point3: torch.Tensor
    ):
  
    """
    Checks if orientation of points is counter-clockwise (1), clockwise (-1) and co-linear (0)
    """
    orientation = (point3[1] - point2[1])*(point2[0] - point1[0]) \
                - (point2[1] - point1[1])*(point3[0] - point2[0])
    # clockwise
    if orientation > 0:
        return 1
    # counter-clockwise
    elif orientation < 0:
        return - 1
    # co-linear
    else:
        return 0
    
def dist(
      point1: torch.Tensor,
      point2: torch.Tensor
    ):
    """
    Calculates the distance between 2 points
    """
    return math.sqrt((point2[1]-point1[1])**2 \
        + (point2[0]-point1[0])**2)


def create_polygon(points: torch.Tensor) -> torch.LongTensor:
    
    """
    Implentation of Jarvis Scan algorithm : Idea is to find the most counter-cloclwise point 
    after adding the left-most point and keep iterating untill we arrive at the first point
    """
    
    # find the leftmost point
    # if two points have same leftmost coordinate
    # find the one with the maximum y-coordinate
    left_most_point =  min(points, key = lambda point: (point[0], -point[1]))
    convex_hull = []

    # hull point represents last point added to hull
    # the left most point is gauranted to be on the convex hull
    hull_point = left_most_point

    # start from the leftmost points and 
    while True:
        convex_hull.append(hull_point.tolist())

        # initialise with the current_element as the first element
        current_point = points[0]
        for next_point in points:
            o = orientation(hull_point, current_point, next_point)

            # start with a point that could potentially be added to the convex hull
            # and find the point which forms the most counter-clockwise angle and is farthest with respect to 
            # most recently added point on convex hull

            # Condition 1: (hull_point, current_point, next_point) should make a counter-clockwise angle
            # If there exists a next_point such that (hull_point, current_point, next_point) is more counter-clockwise 
            # we ignore current_point and update current_point = next_point since including next_point on hull should not automatically include current_point
            if o == 1:
                current_point = next_point

            # Condition 2: Ignore if the point is already added to hull and move to next_point
            if torch.equal(current_point, hull_point):
                current_point = next_point

            # Condition 3: If the set of points (hull_point, current_point, next_point) are collinear, consider the farthest point
            if o == 0 and dist(hull_point, next_point) > dist(hull_point, current_point):
                current_point = next_point

        # current_point forms the most counter-clockwise angle with respect to recent hull_point
        # therefore we add this point to the hull
        hull_point = current_point

        # If we reach from where we started then we exit the loop
        if torch.equal(hull_point, torch.Tensor(convex_hull[0]).long()):
            break
            
    return torch.LongTensor(convex_hull)

def get_tight_polygon_from_mask(test_mask_tensor: torch.Tensor):

  # An algorithm that computes the enclosing polygon from the segmentation mask.
  mask_points_n2 = torch.stack(torch.where(test_mask_tensor == 1), 1)
  polygon_points_n2 = create_polygon(mask_points_n2)
  
  return polygon_points_n2