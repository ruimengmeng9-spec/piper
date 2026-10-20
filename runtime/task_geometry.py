"""Pure geometry for a square through-hole made from four convex walls."""

def hole_geometry(peg_width, hole_width, outer_width=.08, height=.06):
    if not 0 < peg_width < hole_width < outer_width:
        raise ValueError('Require 0 < peg width < hole width < block width')
    clearance = (hole_width-peg_width)/2
    thickness = (outer_width-hole_width)/2
    center = (outer_width+hole_width)/4
    return dict(
        hole_width=hole_width, side_clearance=clearance,
        contact_offset=min(.0003,.1*clearance),
        success_xy_tolerance=min(.0025,.5*clearance),
        containment_margin=min(.0001,.1*clearance),
        wall_offsets=[(center,0.,-height/2),(-center,0.,-height/2),
                      (0.,center,-height/2),(0.,-center,-height/2)],
        wall_sizes=[(thickness,outer_width,height)]*2+[(hole_width,thickness,height)]*2)
