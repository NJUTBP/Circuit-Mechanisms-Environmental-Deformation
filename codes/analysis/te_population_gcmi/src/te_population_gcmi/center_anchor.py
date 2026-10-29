"""Position-only centre-anchor arc-length extension, with disjoint centre-visit budgets.

No neural signals or TE values participate in endpoint selection.
"""
from dataclasses import dataclass
import numpy as np

from .segments import _candidate_intervals, WALL_NAMES, WALL_AXES


@dataclass
class Candidate:
    counter: int
    start: int
    stop: int
    direction: str
    axis: str
    wall: str
    anchor: int
    budget_limit: int
    indices: np.ndarray
    running_arclength: np.ndarray
    base_qc: bool


def prepare_candidates(positions, width, height=1.0, centre_fraction=.6, boundary=.1):
    positions = np.asarray(positions, float)
    mx = width * (1-centre_fraction)/2
    my = height * (1-centre_fraction)/2
    walls = np.c_[positions[:,0], width-positions[:,0], positions[:,1], height-positions[:,1]]
    assert walls.min() > -1e-6
    center = ((positions[:,0]>=mx) & (positions[:,0]<=width-mx)
              & (positions[:,1]>=my) & (positions[:,1]<=height-my))
    boundary_mask = walls.min(1)<=boundary
    assert not np.any(center & boundary_mask)
    labels = np.zeros(len(positions), np.int8)
    labels[center]=1; labels[boundary_mask]=2
    intervals = _candidate_intervals(labels)
    starts = np.flatnonzero(center & ~np.r_[False,center[:-1]])
    stops = np.flatnonzero(center & ~np.r_[center[1:],False])+1
    visit_id = np.full(len(center),-1,int)
    for j,(a,b) in enumerate(zip(starts,stops)):
        visit_id[a:b]=j
    uses = {}
    for a,b,d in intervals:
        anchor = a if d=='approach' else b-1
        uses.setdefault(visit_id[anchor],set()).add(d)
    cumulative = np.r_[0,np.cumsum(np.linalg.norm(np.diff(positions,axis=0),axis=1))]
    result=[]
    for counter,(a,b,d) in enumerate(intervals):
        anchor = a if d=='approach' else b-1
        visit = visit_id[anchor]
        lo,hi=starts[visit],stops[visit]-1
        # Incoming departure and outgoing approach receive disjoint halves of
        # the same centre visit. Split by travelled distance, not closest centre.
        if len(uses[visit])==2 and hi>lo:
            midpoint = int(np.searchsorted(cumulative, (cumulative[lo]+cumulative[hi])/2))
            midpoint = max(lo,min(midpoint,hi-1))
            limit = midpoint+1 if d=='approach' else midpoint
        else:
            limit = lo if d=='approach' else hi
        step = -1 if d=='approach' else 1
        indices = np.arange(anchor,limit+step,step,dtype=int)
        running = np.abs(cumulative[indices]-cumulative[anchor])
        wi=b-1 if d=='approach' else a
        order=np.argsort(walls[wi])
        wall=WALL_NAMES[int(order[0])]
        unambiguous=walls[wi,order[1]]-walls[wi,order[0]]>1e-9
        result.append(Candidate(counter,a,b,d,WALL_AXES[wall],wall,anchor,limit,
                                indices,running,bool(unambiguous)))
    return result,cumulative,labels,walls


def choose_endpoint(candidate, range_m):
    if not np.isfinite(range_m) or range_m<0:
        raise ValueError('range_m must be finite and non-negative')
    if range_m==0:
        return candidate.anchor,False
    # Follow recorded points only, up to the cumulative arc-length budget.
    count=int(np.searchsorted(candidate.running_arclength,range_m,side='right'))
    endpoint=int(candidate.indices[max(0,count-1)])
    reached=count<len(candidate.indices)
    return endpoint,reached


def classify_at_range(candidates,cumulative,walls,range_m,minimum_frames=32):
    rows=[]
    for c in candidates:
        endpoint,reached=choose_endpoint(c,range_m)
        start=endpoint if c.direction=='approach' else c.start
        stop=c.stop if c.direction=='approach' else endpoint+1
        frames=stop-start
        delta=walls[stop-1,WALL_NAMES.index(c.wall)]-walls[start,WALL_NAMES.index(c.wall)]
        net_ok=delta<0 if c.direction=='approach' else delta>0
        rows.append((c.counter,c.direction,c.axis,c.wall,start,stop,frames,
            (frames-1)*.01,(cumulative[stop-1]-cumulative[start])*100,
            c.base_qc and net_ok and frames>=minimum_frames,
            c.anchor,endpoint,c.budget_limit,reached,abs(endpoint-c.anchor)))
    return rows


COLUMNS=['counter','direction','wall_axis','wall','start_frame','stop_frame_exclusive',
         'n_frames','duration_s','path_length_cm','passed_qc','crossing_frame',
         'centre_endpoint_frame','centre_budget_limit','range_reached','added_frames']


def endpoint_extensions(positions, width, height, segmentation):
    """Return optional production interval overrides from a frozen geometry rule."""
    spec=segmentation.get('centre_anchor_extension')
    if spec is None:
        return None
    if spec.get('mode')!='entry_exit_arclength':
        raise ValueError('Unknown centre_anchor_extension mode')
    if not any(np.isclose(width,v,atol=1e-8,rtol=0) for v in spec['allowed_widths_m']):
        raise ValueError('Centre-anchor rule was not calibrated for this arena width')
    candidates,cs,_,walls=prepare_candidates(
        positions,width,height,float(segmentation['centre_fraction']),
        float(segmentation['boundary_zone_m']))
    result=[]
    for c in candidates:
        length=width if c.axis=='x' else height
        limit=(float(spec['base_range_m'])+float(spec['gap_compensation']) *
              (1-float(segmentation['centre_fraction']))/2 *
              (float(spec['reference_length_m'])-length))
        if limit<0:
            raise ValueError('Derived centre-anchor path range is negative')
        endpoint,reached=choose_endpoint(c,limit)
        start=endpoint if c.direction=='approach' else c.start
        stop=c.stop if c.direction=='approach' else endpoint+1
        added=abs(cs[endpoint]-cs[c.anchor])
        assert added<=limit+1e-9
        result.append((start,stop,{
            'original_start_frame':c.start,'original_stop_frame_exclusive':c.stop,
            'centre_crossing_frame':c.anchor,'centre_endpoint_frame':endpoint,
            'centre_budget_limit_frame':c.budget_limit,
            'centre_anchor_range_m':limit,'centre_anchor_added_path_m':added,
            'centre_anchor_range_reached':reached,
            'centre_anchor_policy':'entry_exit_arclength'}))
    return result
