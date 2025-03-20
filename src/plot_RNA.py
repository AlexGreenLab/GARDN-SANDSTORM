"""Module for using the forgi library to make RNA plots
"""
import nupack as n
import matplotlib.pyplot as plt
import numpy as np
import GA_util
import util
import colorsys
import forgi.threedee.utilities.vector as ftuv
import itertools
import forgi
import math
from math import sin, cos, radians, acos, degrees

####Code modified from Forgi Package#####
####https://github.com/ViennaRNA/forgi#####

def _find_annot_pos_on_circle(nt, coords, cg):
    for i in range(5):
        for sign in [-1,1]:
            a = np.pi/4*i*sign
            if cg.get_elem(nt)[0]=="s":
                bp = cg.pairing_partner(nt)
                anchor = coords[bp-1]
            else:
                anchor =np.mean([ coords[nt-2], coords[nt]], axis=0)
            vec = coords[nt-1]-anchor
            vec=vec/ftuv.magnitude(vec)
            rotated_vec =  np.array([vec[0]*math.cos(a)-vec[1]*math.sin(a),
                                     vec[0]*math.sin(a)+vec[1]*math.cos(a)])
            annot_pos = coords[nt-1]+rotated_vec*18
            if _clashfree_annot_pos(annot_pos, coords):
                # log.debug("Annot pos on c is %s",annot_pos)
                return annot_pos
    return None

def circles(x, y, s, c='b', ax=None, vmin=None, vmax=None,labels=[], **kwargs):
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    #import matplotlib.colors as colors

    if ax is None:
        ax = plt.gca()

    if isinstance(c, basestring):
        color = c     # ie. use colors.colorConverter.to_rgba_array(c)
    else:
        color = None  # use cmap, norm after collection is created
    kwargs.update(color=color)

    if np.isscalar(x):
        patches = [Circle((x, y), s), ]
    elif np.isscalar(s):
        patches = [Circle((x_, y_), s) for x_, y_ in zip(x, y)]
    else:
        patches = [Circle((x_, y_), s_) for x_, y_, s_ in zip(x, y, s)]

    collection = PatchCollection(patches, **kwargs)

    if color is None:
        collection.set_array(np.asarray(c))
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)

    ax.add_collection(collection)
    ax.autoscale_view()
    return collection

def _clashfree_annot_pos(pos, coords):
    for c in coords:
        dist = ftuv.vec_distance(c, pos)
        #log.debug("vec_dist=%s", dist)
        if dist<14:
            return False
    return True

def _annotate_rna_plot(ax, cg, coords, annotations, text_kwargs):
    # Plot annotations
    annot_dict = { elem:elem for elem in cg.defines}
    if annotations is None:
        annot_dict={elem:"" for elem in cg.defines}
    else:
        annot_dict.update(annotations)
    stem_coords = {}
    for stem in cg.stem_iterator():
        stem_start =  np.mean([coords[cg.defines[stem][0]-1],
                               coords[cg.defines[stem][3]-1]],
                               axis=0)
        stem_end =  np.mean([coords[cg.defines[stem][1]-1],
                               coords[cg.defines[stem][2]-1]],
                               axis=0)
        stem_center = np.mean([stem_start, stem_end], axis=0)
        stem_coords[stem]=(stem_start, stem_center, stem_end)
        if annot_dict[stem]:
            stem_vec = stem_end - stem_start
            norm_vec = (stem_vec[1], -stem_vec[0])
            norm_vec/=ftuv.magnitude(norm_vec)
            annot_pos = np.array(stem_center)+23*norm_vec
            #log.debug("Checking clashfree for %s, %s", stem, annot_pos)
            if not _clashfree_annot_pos(annot_pos, coords):
                log.debug("Cannot annotate %s as %s ON THE RIGHT HAND SIDE, because of insufficient space. Trying left side...", stem, annot_dict[stem])
                annot_pos = np.array(stem_center)-23*norm_vec
                #log.debug("Checking clashfree OTHER SIDE for %s, %s", stem, annot_pos)
                if not _clashfree_annot_pos(annot_pos, coords):
                    log.info("Cannot annotate %s as '%s', because of insufficient space.", stem, annot_dict[stem])
                    annot_pos = None
            #log.debug("%s", annot_pos)
            if annot_pos is not None:
                ax.annotate(annot_dict[stem], xy=annot_pos,
                            ha="center", va="center", **text_kwargs )
    for hloop in cg.hloop_iterator():
        hc = []
        for nt in cg.define_residue_num_iterator(hloop, adjacent=True):
            hc.append(coords[nt-1])
        annot_pos = np.mean(hc, axis=0)
        if _clashfree_annot_pos(annot_pos, coords):
            ax.annotate(annot_dict[hloop], xy=annot_pos,
                        ha="center", va="center", **text_kwargs )
        else:
            log.info("Cannot annotate %s as '%s' ON THE INSIDE, because of insufficient space. Trying outside...", hloop, annot_dict[hloop])
            nt1, nt2 = cg.define_a(hloop)
            start = np.mean([coords[nt1-1], coords[nt2-1]], axis=0)
            vec = annot_pos-start
            annot_pos = annot_pos+vec*3
            if _clashfree_annot_pos(annot_pos, coords):
                ax.annotate(annot_dict[hloop], xy=annot_pos,
                            ha="center", va="center", **text_kwargs )
            else:
                log.info("Cannot annotate %s as '%s', because of insufficient space.", hloop, annot_dict[hloop])
    for iloop in cg.iloop_iterator():
        s1, s2 = cg.connections(iloop)
        annot_pos = np.mean([ stem_coords[s1][2], stem_coords[s2][0]], axis=0)
        if _clashfree_annot_pos(annot_pos, coords):
            ax.annotate(annot_dict[iloop], xy=annot_pos,
                        ha="center", va="center", **text_kwargs )
        else:
            log.debug("Cannot annotate %s as '%s' ON THE INSIDE, because of insufficient space. Trying outside...", iloop, annot_dict[iloop])
            loop_vec = stem_coords[s2][0] - stem_coords[s1][2]
            norm_vec = (loop_vec[1], -loop_vec[0])
            norm_vec/=ftuv.magnitude(norm_vec)
            annot_pos_p = np.array(annot_pos)+25*norm_vec
            annot_pos_m = np.array(annot_pos)-25*norm_vec
            # iloops can be asymmetric (more nts on one strand.)
            # plot the label on the strand with more nts.
            plus=0
            minus=0
            for nt in cg.define_residue_num_iterator(iloop):
                if ftuv.vec_distance(annot_pos_p, coords[nt-1])<ftuv.vec_distance(annot_pos_m, coords[nt-1]):
                    plus+=1
                else:
                    minus+=1
            if plus>minus:
                if _clashfree_annot_pos(annot_pos_p, coords):
                    ax.annotate(annot_dict[iloop], xy=annot_pos_p,
                                ha="center", va="center", **text_kwargs )
                else:
                    log.info("Cannot annotate %s as '%s' (only trying inside and right side), because of insufficient space.", iloop, annot_dict[iloop])

            else:
                if _clashfree_annot_pos(annot_pos_m, coords):
                    ax.annotate(annot_dict[iloop], xy=annot_pos_m,
                                ha="center", va="center", **text_kwargs )
                else:
                    log.info("Cannot annotate %s as '%s' (only trying inside and left side), because of insufficient space.", iloop, annot_dict[iloop])
    for mloop in itertools.chain(cg.floop_iterator(), cg.tloop_iterator(), cg.mloop_iterator()):
        nt1, nt2 = cg.define_a(mloop)
        res = list(cg.define_residue_num_iterator(mloop))
        if len(res)==0:
            anchor = np.mean([coords[nt1-1], coords[nt2-1]], axis=0)
        elif len(res)%2==1:
            anchor = coords[res[int(len(res)//2)]-1]
        else:
            anchor =  np.mean([ coords[res[int(len(res)//2)-1]-1],
                                coords[res[int(len(res)//2)]-1] ],
                              axis=0)
        loop_vec = coords[nt1-1] - coords[nt2-1]
        norm_vec = (loop_vec[1], -loop_vec[0])
        norm_vec/=ftuv.magnitude(norm_vec)
        annot_pos = anchor - norm_vec*18
        if _clashfree_annot_pos(annot_pos, coords):
            ax.annotate(annot_dict[mloop], xy=annot_pos,
                        ha="center", va="center", **text_kwargs )
        else:
            log.info("Cannot annotate %s as '%s' , because of insufficient space.", mloop, annot_dict[mloop])
            
            






#-----------------------------------------------
def plot_rna(cg, ax=None, offset=(0, 0), text_kwargs={}, backbone_kwargs={},
             basepair_kwargs={}, color=True, lighten=0, annotations={},alpha=1,max_likelihood=False):
    
    #Max likelihood option will plot the structure with black edge colors

    import RNA
    import matplotlib.colors as mc
    RNA.cvar.rna_plot_type = 1

    coords = []
    #colors = []
    #circles = []

    bp_string = cg.to_dotbracket_string()
    el_string = cg.to_element_string()
    el_to_color = {'f': 'orange',
                   't': 'orange',
                   's': 'green',
                   'h': 'blue',
                   'i': 'yellow',
                   'm': 'red'}

    if ax is None:
        ax = plt.gca()

    if offset is None:
        offset = (0, 0)
    elif offset is True:
        offset = (ax.get_xlim()[1], ax.get_ylim()[1])
    else:
        pass

    #Standard Forgi Coordinates
    vrna_coords = RNA.get_xy_coordinates(bp_string)
    for i, _ in enumerate(bp_string):
        coord = (offset[0] + vrna_coords.get(i).X,
                 offset[1] + vrna_coords.get(i).Y)
        coords.append(coord)     
    coords = np.array(coords)

    
    
    # coords = RNApairs2coordinates(RNA.ptable(bp_string))

    coords = coords - coords.mean(axis=0)
    coords[:,1]*=-1
    coords = rotate_coords(coords)
    # centroid = np.mean(coords)
    # print(centroid)
    

    # plt.scatter(0,0,marker='*')

    # First plot backbone
    bkwargs = {"color":"black", "zorder":0}
    bkwargs.update(backbone_kwargs)
    ax.plot(coords[:,0], coords[:,1], **bkwargs,alpha=alpha)
    # ax.set_xlim([-np.max(coords),np.max(coords)])
    # ax.set_ylim([-np.max(coords),np.max(coords)])

    # Now plot basepairs
    basepairs = []
    for s in cg.stem_iterator():
        for p1, p2 in cg.stem_bp_iterator(s):
            basepairs.append([coords[p1-1], coords[p2-1]])
    if basepairs:
        basepairs = np.array(basepairs)
        if color:
            c = "red"
        else:
            c = "black"
            bpkwargs = {"color":c, "zorder":0, "linewidth":3}
            bpkwargs.update(basepair_kwargs)
            ax.plot(basepairs[:,:,0].T, basepairs[:,:,1].T,lw=2 **bpkwargs)
#     # Now plot circles

    if max_likelihood==False:
        for i, coord in enumerate(coords):
            if color:
                c = el_to_color[el_string[i]]
                h,l,s = colorsys.rgb_to_hls(*mc.to_rgb(c))
                if lighten>0:
                    l += (1-l)*min(1,lighten)
                else:
                    l +=l*max(-1, lighten)
                if l>1 or l<0:
                    print(l)
                c=colorsys.hls_to_rgb(h,l,s)
                circle = plt.Circle((coord[0], coord[1]),
                                color=c,alpha=alpha)
            else:
                circle = plt.Circle((coord[0], coord[1]),
                                    edgecolor="black", facecolor="white",alpha=alpha)

            ax.add_artist(circle)
    else:
        for i, coord in enumerate(coords):
            if color:
                c = el_to_color[el_string[i]]
                h,l,s = colorsys.rgb_to_hls(*mc.to_rgb(c))
                if lighten>0:
                    l += (1-l)*min(1,lighten)
                else:
                    l +=l*max(-1, lighten)
                if l>1 or l<0:
                    print(l)
                c=colorsys.hls_to_rgb(h,l,s)
                circle = plt.Circle((coord[0], coord[1]),
                                facecolor=c,alpha=1,edgecolor="black")
            else:
                circle = plt.Circle((coord[0], coord[1]),
                                    edgecolor="black", facecolor="white",alpha=alpha)

            ax.add_artist(circle)
            ax.axis('equal')
        

    ax.set_axis_off()

    return (ax, coords)



def rotate_point(point, angle, center_point=(0, 0)):
    """Rotates a point around center_point(origin by default)
    Angle is in degrees.
    Rotation is counter-clockwise
    """
    angle_rad = radians(angle % 360)
    # Shift the point so that center_point becomes the origin
    new_point = (point[0] - center_point[0], point[1] - center_point[1])
    new_point = (new_point[0] * cos(angle_rad) - new_point[1] * sin(angle_rad),
                 new_point[0] * sin(angle_rad) + new_point[1] * cos(angle_rad))
    # Reverse the shifting we have done
    new_point = (new_point[0] + center_point[0], new_point[1] + center_point[1])
    return new_point


import numpy as np

def rotate_coords(coords):
    #return rotated coordinates

    try:
        for stem in cg.stem_iterator():
            stem_start =  np.mean([coords[cg.defines[stem][0]-1],
                               coords[cg.defines[stem][3]-1]],
                               axis=0)
            stem_end =  np.mean([coords[cg.defines[stem][1]-1],
                               coords[cg.defines[stem][2]-1]],
                               axis=0)
            stem_center = np.mean([stem_start, stem_end], axis=0)
    except:
        stem_center = 0
    a = np.array([np.min(coords[:,0]),np.max(coords[:,1])])
    b = np.array([stem_center,stem_center])
    c = np.array([0,1])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    ANGLE = np.arccos(cosine_angle)
    ANGLE = np.degrees(ANGLE) - 2.5
    

    
    
    for i in range(coords.shape[0]):
        coords[i,:] = rotate_point(coords[i,:],angle=ANGLE,center_point=(stem_center,stem_center))
    
    return coords



import forgi.visual.mplotlib as fvm

def return_secondary_structure_plots(sequences,alpha=0.04,title='Secondary Structure Distribution',max_likelihood=False,figsize=(4,3)):
    #Sequences should be one_hot_encoded and shape n_seqs,4,seq_len 
    mod = n.Model()
    plt.figure(figsize=figsize)
    for i in range(sequences.shape[0]):
        seq_nts = GA_util.unencode(sequences[i,:,:])
        structure = str(GA_util.calc_struc(sequences[i,:,:],mod))
        
        forgi_obj = forgi.graph.bulge_graph.BulgeGraph.from_dotbracket(structure,seq_nts)
        
        plot_rna(forgi_obj,lighten=0.2,annotations=None,alpha=alpha,max_likelihood=False)

        
    if max_likelihood==True:
        output = GA_util.calc_consensus_structure(sequences)

        z = np.argmax(output,axis=0)
    
        #Decoding back to dot parens notation
        consensus = ''
        for i in range(z.shape[0]):
        
            # prob *= output[z[i],i]
            if z[i] == 0:
                consensus += '.'
            elif z[i] == 1:
                consensus += '('
            elif z[i] == 2:
                consensus += ')'
        try:
            if str(consensus) == '.'*len(consensus):
                ml_forgi_obj = forgi.graph.bulge_graph.BulgeGraph.from_dotbracket(consensus)
                (ax,coords) = plot_rna(ml_forgi_obj,lighten=0.2,annotations=None,alpha=alpha,max_likelihood=True)
            else:
                ml_forgi_obj = forgi.graph.bulge_graph.BulgeGraph.from_dotbracket(consensus)
                (ax,coords) = plot_rna(ml_forgi_obj,lighten=0.2,annotations=None,alpha=alpha,max_likelihood=True)
                ax.set_xlim([-np.max(coords),np.max(coords)])

        except:
            ml_forgi_obj = forgi.graph.bulge_graph.BulgeGraph.from_dotbracket('.'*sequences.shape[2])
            (ax,coords) = plot_rna(ml_forgi_obj,lighten=0.2,annotations=None,alpha=alpha,max_likelihood=True)
            # ax.autoscale()

            
            
    

    plt.title('%s'%title)