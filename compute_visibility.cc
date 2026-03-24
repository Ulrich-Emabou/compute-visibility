/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */
#include <chrono>    // For std::chrono
#include <thread>    // For std::this_thread
#include <mutex>     // For std::mutex, std::lock_guard
#include <cassert>   // For assert macro (at line 282)
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <string>
#include <cstring>
#include <cerrno>
#include <cstdlib>

#include <octomap/octomap_utils.h>
#include "util/system.h"
#include "util/arguments.h"
#include "util/tokenizer.h"
#include "mve/depthmap.h"
#include "mve/mesh_info.h"
#include "mve/mesh_io.h"
#include "mve/mesh_io_ply.h"
#include "mve/mesh_tools.h"
#include "mve/scene.h"
#include "mve/view.h"

#include <armadillo>

#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap/Pointcloud.h>

#include <embree2/rtcore.h>
#include <embree2/rtcore_scene.h>
#include <embree2/rtcore_ray.h>

struct Vertex   { float x, y, z, a; };
struct Triangle { int v0, v1, v2; };

int
compute_visibility_unoriented_point_unoriented_point(arma::vec pt_1, arma::vec pt_2, RTCScene rtc_scene, float eps=1e-6)
{
    // pt_1_to_pt_2      = pt_2 - pt_1
    // dist_pt_1_to_pt_2 = norm(pt_1_to_pt_2)
    // ray               = Ray_3(pt_1_cgal, pt_2_cgal)
    // intersections     = []

    // geometry_aabb_tree_cgal.all_intersections(ray, intersections)
    
    // if len(intersections) == 0:
    //     return 1
    
    // intersections_point_3_cgal = []

    // for i in intersections:
    //     if i.first.is_Point_3():
    //         intersections_point_3_cgal.append(i.first.get_Point_3())
    //     elif i.first.is_Segment_3():
    //         assert i.first.get_Segment_3().is_degenerate()
    //         intersections_point_3_cgal.append(i.first.get_Segment_3().point(0))
    //     else:
    //         assert False
    
    // dists_unsorted = array([ sqrt((i - pt_1_cgal).squared_length()) for i in intersections_point_3_cgal ])

    // if any(logical_and(dists_unsorted > eps, dists_unsorted < dist_pt_1_to_pt_2 - eps)):
    //     return 0

    // return 1

    arma::vec pt_1_to_pt_2            = pt_2 - pt_1;
    arma::vec pt_1_to_pt_2_normalized = normalise(pt_1_to_pt_2);

    // std::cout << pt_1 << std::endl;
    // std::cout << pt_2 << std::endl;
    // std::cout << pt_1_to_pt_2 << std::endl;
    // std::cout << std::endl;

    float dist_pt_1_to_pt_2 = arma::norm(pt_1_to_pt_2);

    // std::cout << dist_pt_1_to_pt_2 << std::endl;
    // std::cout << std::endl;

    /* initialize ray */
    RTCRay ray;
    ray.org[0] = (float)pt_1(0);
    ray.org[1] = (float)pt_1(1);
    ray.org[2] = (float)pt_1(2);
    ray.dir[0] = (float)pt_1_to_pt_2_normalized(0);
    ray.dir[1] = (float)pt_1_to_pt_2_normalized(1);
    ray.dir[2] = (float)pt_1_to_pt_2_normalized(2);
    ray.tnear  = eps;
    ray.tfar   = 999999.0f;
    ray.time   = 0;
    ray.mask   = -1;
    ray.geomID = RTC_INVALID_GEOMETRY_ID;
    ray.primID = RTC_INVALID_GEOMETRY_ID;
    ray.instID = RTC_INVALID_GEOMETRY_ID;

    // std::cout << ray.org[0] << std::endl;
    // std::cout << ray.org[1] << std::endl;
    // std::cout << ray.org[2] << std::endl;
    // std::cout << ray.dir[0] << std::endl;
    // std::cout << ray.dir[1] << std::endl;
    // std::cout << ray.dir[2] << std::endl;
    // std::cout << ray.tnear  << std::endl;
    // std::cout << ray.tfar   << std::endl;
    // std::cout << ray.time   << std::endl;
    // std::cout << ray.mask   << std::endl;
    // std::cout << ray.geomID << std::endl;
    // std::cout << ray.primID << std::endl;
    // std::cout << ray.instID << std::endl;
    // std::cout << std::endl;

    /* intersect ray with scene */
    rtcIntersect(rtc_scene, ray);

    if (ray.tfar != 999999.0f)
    {
        assert(ray.tfar >= eps);
        assert(ray.geomID != RTC_INVALID_GEOMETRY_ID);
        assert(ray.primID != RTC_INVALID_GEOMETRY_ID);
    }
    else
    {
        assert(ray.geomID == RTC_INVALID_GEOMETRY_ID);
        assert(ray.primID == RTC_INVALID_GEOMETRY_ID);
    }

    if (ray.tfar > eps && ray.tfar < dist_pt_1_to_pt_2 - eps)
    {
        // std::cout << "0 " << ray.tfar << " " << ray.geomID << " " << ray.primID << std::endl;

        return 0;
    }
    else
    {
        // std::cout << "1 " << ray.tfar << " " << ray.geomID << " " << ray.primID << std::endl;

        return 1;
    }
}

int
compute_visibility_unoriented_point_oriented_point(arma::vec pt_1, arma::vec pt_2, arma::vec pt_2_normal, float pts_2_cone_half_angle, RTCScene rtc_scene, float eps=1e-6)
{
    // if arccos(dot(sklearn.preprocessing.normalize([pt_1 - pt_2])[0],
    //               sklearn.preprocessing.normalize([pt_2_normal])[0])) >= pts_2_cone_half_angle:
    //     return 0
    
    // return compute_visibility_unoriented_point_unoriented_point(pt_1, pt_2, pt_1_cgal, pt_2_cgal, geometry_aabb_tree_cgal, eps)

    if (acos(arma::norm_dot((pt_1 - pt_2).t(), pt_2_normal)) >= pts_2_cone_half_angle)
    {
        return 0;
    }

    return compute_visibility_unoriented_point_unoriented_point(pt_1, pt_2, rtc_scene, eps);
}

int
compute_visibility_oriented_point_unoriented_point(arma::vec pt_1, arma::vec pt_1_normal, float pts_1_cone_half_angle, arma::vec pt_2, RTCScene rtc_scene, float eps=1e-6)
{
    // if arccos(dot(sklearn.preprocessing.normalize([pt_2 - pt_1])[0],
    //               sklearn.preprocessing.normalize([pt_1_normal])[0])) >= pts_1_cone_half_angle:
    //     return 0
    
    // return compute_visibility_unoriented_point_unoriented_point(pt_1, pt_2, pt_1_cgal, pt_2_cgal, geometry_aabb_tree_cgal, eps)

    if (acos(arma::norm_dot((pt_2 - pt_1).t(), pt_1_normal)) >= pts_1_cone_half_angle)
    {
        // std::cout << "early exit..." << std::endl;
        return 0;
    }

    // std::cout << "doing raycast test..." << std::endl;
    return compute_visibility_unoriented_point_unoriented_point(pt_1, pt_2, rtc_scene, eps);
}

int
compute_visibility_oriented_point_oriented_point(arma::vec pt_1, arma::vec pt_1_normal, float pts_1_cone_half_angle, arma::vec pt_2, arma::vec pt_2_normal, float pts_2_cone_half_angle, RTCScene rtc_scene, float eps=1e-6)
{
    // if arccos(dot(sklearn.preprocessing.normalize([pt_1 - pt_2])[0],
    //               sklearn.preprocessing.normalize([pt_2_normal])[0])) >= pts_2_cone_half_angle:
    //     return 0

    // if arccos(dot(sklearn.preprocessing.normalize([pt_2 - pt_1])[0],
    //               sklearn.preprocessing.normalize([pt_1_normal])[0])) >= pts_1_cone_half_angle:
    //     return 0

    // return compute_visibility_unoriented_point_unoriented_point(pt_1, pt_2, pt_1_cgal, pt_2_cgal, geometry_aabb_tree_cgal, eps)

    if (acos(arma::norm_dot((pt_1 - pt_2).t(), pt_2_normal)) >= pts_2_cone_half_angle)
    {
        return 0;
    }

    if (acos(arma::norm_dot((pt_2 - pt_1).t(), pt_1_normal)) >= pts_1_cone_half_angle)
    {
        return 0;
    }

    return compute_visibility_unoriented_point_unoriented_point(pt_1, pt_2, rtc_scene, eps);
}

int
main (int argc, char** argv)
{
    /* Setup argument parser. */

    // def _compute_visibility(pts_1, pts_1_normals, pts_2, pts_2_normals, vertices, faces, pts_1_oriented, pts_1_cone_half_angle, pts_2_oriented, pts_2_cone_half_angle, compute_mask, eps):

    util::Arguments args;
    args.set_exit_on_error(true);
    args.set_nonopt_minnum(13);
    args.set_nonopt_maxnum(13);
    args.set_helptext_indent(25);
    args.set_usage("Usage: compute_visibility [ OPTS ]  PTS_1  PTS_1_NORMALS  PTS_2  PTS_2_NORMALS  VERTICES  INDICES  PTS_1_ORIENTED  PTS_1_CONE_HALF_ANGLE  PTS_2_ORIENTED  PTS_2_CONE_HALF_ANGLE  COMPUTE_MASK  EPS  VISIBILITY_MATRIX_OUT");
    args.parse(argc, argv);

    /* Init default settings. */
    std::string pts_1_in              = args.get_nth_nonopt(0);
    std::string pts_1_normals_in      = args.get_nth_nonopt(1);
    std::string pts_2_in              = args.get_nth_nonopt(2);
    std::string pts_2_normals_in      = args.get_nth_nonopt(3);
    std::string vertices_in           = args.get_nth_nonopt(4);
    std::string indices_in            = args.get_nth_nonopt(5);
    int         pts_1_oriented        = args.get_nth_nonopt_as<int>(6);
    float       pts_1_cone_half_angle = args.get_nth_nonopt_as<float>(7);
    int         pts_2_oriented        = args.get_nth_nonopt_as<int>(8);
    float       pts_2_cone_half_angle = args.get_nth_nonopt_as<float>(9);
    std::string compute_mask_in       = args.get_nth_nonopt(10);
    float       eps                   = args.get_nth_nonopt_as<float>(11);
    std::string visibility_matrix_out = args.get_nth_nonopt(12);



    /* Load data. */
    arma::mat pts_1, pts_1_normals, pts_2, pts_2_normals, vertices, indices, compute_mask;

    pts_1.load(pts_1_in, arma::hdf5_binary_trans);

    if (pts_1_oriented)
    {
        pts_1_normals.load(pts_1_normals_in, arma::hdf5_binary_trans);
    }

    pts_2.load(pts_2_in, arma::hdf5_binary_trans);

    if (pts_2_oriented)
    {
        pts_2_normals.load(pts_2_normals_in, arma::hdf5_binary_trans);
    }

    vertices.load(vertices_in, arma::hdf5_binary_trans);
    indices.load(indices_in, arma::hdf5_binary_trans);

    if (compute_mask_in != "None")
    {
        compute_mask.load(compute_mask_in, arma::hdf5_binary_trans);
    }



    /* Construct RTC data. */

    // vertices_cgal           = [ Point_3(p[0], p[1], p[2]) for p in vertices.astype(float64) ]
    // faces_cgal              = [ Triangle_3(vertices_cgal[f[0]], vertices_cgal[f[1]], vertices_cgal[f[2]]) for f in faces ]
    // geometry_aabb_tree_cgal = AABB_tree_Triangle_3_soup(faces_cgal)

    // pts_1_cgal = [ Point_3(p[0], p[1], p[2]) for p in pts_1.astype(float64) ]
    // pts_2_cgal = [ Point_3(p[0], p[1], p[2]) for p in pts_2.astype(float64) ]

    RTCDevice rtc_device = rtcNewDevice(NULL);
    assert(rtcDeviceGetError(NULL) == RTC_NO_ERROR);

    RTCScene rtc_scene = rtcDeviceNewScene(rtc_device, RTC_SCENE_STATIC, RTC_INTERSECT1);
    assert(rtcDeviceGetError(rtc_device) == RTC_NO_ERROR);

    int rtc_num_triangles = indices.n_rows;
    int rtc_num_vertices  = vertices.n_rows;

    unsigned rtc_triangle_mesh = rtcNewTriangleMesh(rtc_scene, RTC_GEOMETRY_STATIC, rtc_num_triangles, rtc_num_vertices);
    assert(rtcDeviceGetError(rtc_device) == RTC_NO_ERROR);

    /* set vertices */
    Vertex* rtc_vertices = (Vertex*) rtcMapBuffer(rtc_scene, rtc_triangle_mesh, RTC_VERTEX_BUFFER);
    for (int i = 0; i < vertices.n_rows; i++)
    {
        rtc_vertices[i].x = vertices(i,0);
        rtc_vertices[i].y = vertices(i,1);
        rtc_vertices[i].z = vertices(i,2);
    }
    rtcUnmapBuffer(rtc_scene, rtc_triangle_mesh, RTC_VERTEX_BUFFER);

    /* set triangles */
    Triangle* rtc_triangles = (Triangle*) rtcMapBuffer(rtc_scene, rtc_triangle_mesh, RTC_INDEX_BUFFER);
    for (int i = 0; i < indices.n_rows; i++)
    {
        rtc_triangles[i].v0 = indices(i,0);
        rtc_triangles[i].v1 = indices(i,1);
        rtc_triangles[i].v2 = indices(i,2);
    }
    rtcUnmapBuffer(rtc_scene, rtc_triangle_mesh, RTC_INDEX_BUFFER);

    rtcCommit(rtc_scene);



    // num_pts_1 = pts_1.shape[0]
    // num_pts_2 = pts_2.shape[0]

    int num_pts_1 = pts_1.n_rows;
    int num_pts_2 = pts_2.n_rows;

    // visibility_matrix = zeros((num_pts_1,num_pts_2), dtype=int32)

    arma::mat visibility_matrix = arma::mat(num_pts_1, num_pts_2, arma::fill::zeros);

    // if compute_mask is None:
    //     compute_mask = ones_like(visibility_matrix)

    if (compute_mask_in == "None")
    {
        compute_mask = arma::mat(num_pts_1, num_pts_2, arma::fill::ones);
    }

    // compute_coords     = c_[ where(compute_mask)[0], where(compute_mask)[1] ]
    // num_compute_coords = compute_coords.shape[0]

    // for i,d in debug_itertools.range(num_compute_coords):
        
    //     if d: print "    %0.2f%%" % (100.0*float(i) / float(num_compute_coords-1))

    //     p1i = compute_coords[i,0]
    //     p2i = compute_coords[i,1]
        
    //     pt_1      = pts_1[p1i]
    //     pt_1_cgal = pts_1_cgal[p1i]

    //     if pts_1_oriented:
    //         pt_1_normal = pts_1_normals[p1i]

    //     pt_2          = pts_2[p2i]
    //     pt_2_cgal     = pts_2_cgal[p2i]

    //     if pts_2_oriented:
    //         pt_2_normal = pts_2_normals[p2i]

    //     if not pts_1_oriented and not pts_2_oriented:
    //         visibility_matrix[p1i,p2i] = \
    //             compute_visibility_unoriented_point_unoriented_point(pt_1,
    //                                                                  pt_2,
    //                                                                  pt_1_cgal,
    //                                                                  pt_2_cgal,
    //                                                                  geometry_aabb_tree_cgal,
    //                                                                  eps)

    //     if not pts_1_oriented and pts_2_oriented:
    //         visibility_matrix[p1i,p2i] = \
    //             compute_visibility_unoriented_point_oriented_point(pt_1,
    //                                                                pt_2,
    //                                                                pt_2_normal,
    //                                                                pts_2_cone_half_angle,
    //                                                                pt_1_cgal,
    //                                                                pt_2_cgal,
    //                                                                geometry_aabb_tree_cgal,
    //                                                                eps)

    //     if pts_1_oriented and not pts_2_oriented:
    //         visibility_matrix[p1i,p2i] = \
    //             compute_visibility_oriented_point_unoriented_point(pt_1,
    //                                                                pt_1_normal,
    //                                                                pts_1_cone_half_angle,
    //                                                                pt_2,
    //                                                                pt_1_cgal,
    //                                                                pt_2_cgal,
    //                                                                geometry_aabb_tree_cgal,
    //                                                                eps)

    //     if pts_1_oriented and pts_2_oriented:
    //         visibility_matrix[p1i,p2i] = \
    //             compute_visibility_oriented_point_oriented_point(pt_1,
    //                                                              pt_1_normal,
    //                                                              pts_1_cone_half_angle,
    //                                                              pt_2,
    //                                                              pt_2_normal,
    //                                                              pts_2_cone_half_angle,
    //                                                              pt_1_cgal,
    //                                                              pt_2_cgal,
    //                                                              geometry_aabb_tree_cgal,
    //                                                              eps)

    // for (int j = 887; j < 888; j++)
    for (int j = 0; j < num_pts_1; j++)
    {
        if (j % 100 == 0)
        {
            std::cout << j << std::endl;
        }

        for (int i = 0; i < num_pts_2; i++)
        // for (int i = 55; i < 56; i++)
        {
            if (compute_mask(j,i) == 1)
            {
                arma::vec pt_1 = pts_1.row(j).t();
                arma::vec pt_2 = pts_2.row(i).t();

                arma::vec pt_1_normal;

                if (pts_1_oriented)
                {
                    pt_1_normal = pts_1_normals.row(j).t();
                }

                arma::vec pt_2_normal;

                if (pts_2_oriented)
                {
                    pt_2_normal = pts_2_normals.row(i).t();
                }

                if (!pts_1_oriented && !pts_2_oriented)
                {
                    visibility_matrix(j,i) = compute_visibility_unoriented_point_unoriented_point(pt_1, pt_2, rtc_scene, eps);
                }

                if (!pts_1_oriented && pts_2_oriented)
                {
                    visibility_matrix(j,i) = compute_visibility_unoriented_point_oriented_point(pt_1, pt_2, pt_2_normal, pts_2_cone_half_angle, rtc_scene, eps);
                }

                if (pts_1_oriented && !pts_2_oriented)
                {
                    // std::cout << "compute_visibility_oriented_point_unoriented_point..." << std::endl;

                    visibility_matrix(j,i) = compute_visibility_oriented_point_unoriented_point(pt_1, pt_1_normal, pts_1_cone_half_angle, pt_2,rtc_scene, eps);
                }

                if (pts_1_oriented && pts_2_oriented)
                {
                    visibility_matrix(j,i) = compute_visibility_oriented_point_oriented_point(pt_1, pt_1_normal, pts_1_cone_half_angle, pt_2, pt_2_normal, pts_2_cone_half_angle, rtc_scene, eps);
                }
            }
        }
    }

    visibility_matrix.save(visibility_matrix_out, arma::hdf5_binary_trans);



    rtcDeleteScene(rtc_scene);
    assert(rtcDeviceGetError(rtc_device) == RTC_NO_ERROR);

    rtcDeleteDevice(rtc_device);
    assert(rtcDeviceGetError(NULL) == RTC_NO_ERROR);



    return EXIT_SUCCESS;
}
