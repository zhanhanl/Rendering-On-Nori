//
// Created by baconleung on 8/27/17.
//

#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/bsdf.h>
#include <nori/sampler.h>
#include <nori/emitter.h>

#include <nori/nanoflann.hpp>
#include <ctime>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace nanoflann;
NORI_NAMESPACE_BEGIN

    class PhotonMappingIntegrator : public Integrator {
    public:

        enum EPhotonMap{
            ECausticPhotonMap = 0,
            EGlobalPhotonMap
        };

        struct PointCloud {
            struct Point
            {
                float  x,y,z;
                Color3f power;
                Vector3f wi;
            };

            std::vector<Point>  pts;

            // Must return the number of data points
            inline size_t kdtree_get_point_count() const { return pts.size(); }

            // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
            inline float kdtree_distance(const float *p1, const size_t idx_p2,size_t /*size*/) const
            {
                const float d0=p1[0]-pts[idx_p2].x;
                const float d1=p1[1]-pts[idx_p2].y;
                const float d2=p1[2]-pts[idx_p2].z;
                return std::sqrt(d0*d0+d1*d1+d2*d2);
            }

            // Returns the dim'th component of the idx'th point in the class:
            // Since this is inlined and the "dim" argument is typically an immediate value, the
            //  "if/else's" are actually solved at compile time.
            inline float kdtree_get_pt(const size_t idx, int dim) const
            {
                if (dim==0) return pts[idx].x;
                else if (dim==1) return pts[idx].y;
                else return pts[idx].z;
            }

            // Optional bounding-box computation: return false to default to a standard bbox computation loop.
            //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
            //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
            template <class BBOX>
            bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

        };

        // construct a kd-tree index:
        typedef KDTreeSingleIndexAdaptor<
                L2_Simple_Adaptor<float, PointCloud> ,
                PointCloud,
                3 /* dim */
        > my_kd_tree_t;

        PointCloud cloud, caustic_cloud;
        my_kd_tree_t  index, caustic_index; //(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );

        void preprocess(const Scene *scene) {
            // Randomize Seed
            cout<<"photon_mapping..."<<endl;
            std::cout<<"Generating "<< max_photon_count << " photons for Global Photon Map"<< std::endl;
            std::cout<<"Generating "<< caustic_max_photon_count << " photons for Caustic Photon Map"<< std::endl;
            int total_count = 0;

            {
                std::unique_ptr<Sampler> sampler(scene->getSampler()->clone());
                float totalArea = 0;

                /* collect light sources */
                std::vector<Mesh *> m_light_src;
                for (Mesh *mesh:scene->getMeshes()) {
                    if (mesh->isEmitter()) {
                        m_light_src.push_back(mesh);
                    }
                }

                /* sum up the area of all light sources */
                for(Mesh *mesh:m_light_src){
                    totalArea += mesh->getEmitter()->getArea();
                }

                /*
                 * build the GLOBAL PHOTON MAP
                 * */
                for(Mesh *mesh:m_light_src){
                    Emitter *emitter = mesh->getEmitter();
                    int photon_count = max_photon_count * emitter->getArea() / totalArea;
                    int n_emitted = 0;
                    float invPPickLight = totalArea / emitter->getArea();

                    // THETA = pi * A * Le
                    Color3f photon_power = emitter->getEnergy() * emitter->getArea() * M_PI * invPPickLight;

                    while(n_emitted++ < photon_count){
                        Intersection x,y;
                        Point3f y0_p;
                        Normal3f y0_n;

                        /* sample a point on the light source */
                        emitter->samplePosition(sampler->next1D(), sampler->next2D(), y0_p, y0_n);
                        x.p = y0_p;
                        x.mesh = mesh;
                        x.shFrame = Frame(y0_n);

                        /* photon tracing */
                        int depth = 5;
                        float SRVL = 0.9f, pSRVL;
                        Color3f power = photon_power, throughput = Color3f(1.0f);
                        Vector3f wi = y0_n;
                        BSDFQueryRecord bRec = BSDFQueryRecord(x.toLocal(wi));
                        bRec.measure = ESolidAngle;
                        const BSDF *bsdf = nullptr;
                        bool sampledFromEmitter = true;

                        while(true){


                            bRec.wi = x.toLocal(wi);
                            bsdf = x.mesh->getBSDF();

                            if(sampledFromEmitter){
                                bsdf->sample(bRec, sampler->next2D());
                                sampledFromEmitter = false;
                            }else {
                                throughput *= bsdf->sample(bRec, sampler->next2D());
                            }


                            if (!scene->rayIntersect(Ray3f(x.p, -(wi = -x.toWorld(bRec.wo))), y)) {
                                break;
                            }

                            /* add photon to PhotonMap*/
                            if(y.mesh->getBSDF()->isDiffuse()){
                                PointCloud::Point photon;
                                photon.x = y.p.x();
                                photon.y = y.p.y();
                                photon.z = y.p.z();
                                photon.wi = wi;
                                photon.power = power * throughput;
                                cloud.pts.push_back(photon);
                            }

                            x = y;

                            /* depth and the Russian roulette */
                            if(depth-- > 0) pSRVL = 1;
                            else if(sampler->next1D() > SRVL) break;
                            else pSRVL = SRVL;
                            throughput /= pSRVL;

                        }
                    } /* while(n_emitted++ < photon_count) */
                }     /* for(Mesh *mesh:m_light_src) */


                /*
                 * build the CAUSTIC PHOTON MAP
                 */
                for(Mesh *mesh:m_light_src){
                    Emitter *emitter = mesh->getEmitter();
                    int caustic_photon_count = caustic_max_photon_count * emitter->getArea() / totalArea;
                    int n_emitted = 0;

                    // POWER = pi * A * Le
                    Color3f power = emitter->getEnergy() * emitter->getArea() * M_PI;

                    while(n_emitted < caustic_photon_count){

                        Intersection x,y;
                        Point3f  y0_p;
                        Normal3f y0_n;

                        emitter->samplePosition(sampler->next1D(), sampler->next2D(), y0_p, y0_n);
                        x.p = y0_p;
                        x.mesh = mesh;
                        x.shFrame = Frame(y0_n);
                        Vector3f x_wi = y0_n;
                        BSDFQueryRecord x_bRec = BSDFQueryRecord(x.toLocal(x_wi));
                        const BSDF *x_BSDF = x.mesh->getBSDF();

                        // find a specular surface
                        x_BSDF->sample(x_bRec, sampler->next2D());
                        if(!scene->rayIntersect(Ray3f(y0_p, - (x_wi = -x.toWorld(x_bRec.wo))),y)
                                || y.mesh->getBSDF()->isDiffuse()){
                            continue;
                        }

                        x = y;
                        x_bRec.wi = x.toLocal(x_wi);
                        x_BSDF = x.mesh->getBSDF();

                        bool causticFound = true;
                        int  depth = 5;
                        float SRVL = 0.9f;
                        while(!x_BSDF->isDiffuse()){
                            x_BSDF->sample(x_bRec, sampler->next2D());
                            if(!scene->rayIntersect(Ray3f(x.p, -(x_wi = -x.toWorld(x_bRec.wo))), y)){
                                causticFound = false;
                                break;
                            }
                            x = y;
                            x_bRec.wi = x.toLocal(x_wi);
                            x_BSDF = x.mesh->getBSDF();

                            if(depth-- < 0) {
                                causticFound = false;
                                break;
                            }
                        }

                        /* save to the caustic photon map */
                        if(causticFound) {
                            PointCloud::Point photon;
                            photon.x = x.p.x();
                            photon.y = x.p.y();
                            photon.z = x.p.z();
                            photon.wi = x_wi;
                            photon.power = power;
                            caustic_cloud.pts.push_back(photon);
                        }
                    } /* while(n_emitted++ < caustic_photon_count) */
                } /* for(Mesh *mesh:m_light_src) */
            }

            cout<<"cloud's size"<<cloud.pts.size()<<endl;
            cout<<"caustic cloud's size"<<caustic_cloud.pts.size()<<endl;
            std::cout<<"done\n"<<std::endl;

            /* kd-tree */
            index.buildIndex();
            caustic_index.buildIndex();
        }

        PhotonMappingIntegrator(const PropertyList &props)
                : index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */)),
                  caustic_index(3 /*dim*/, caustic_cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */)){

            max_photon_count = props.getInteger("photon_count", 10000);
            rad_estimation_count = props.getInteger("rad_estimation_count", 100);
            rad_estimation_radius = props.getFloat("rad_estimation_radius", 0.05);
            caustic_max_photon_count = props.getInteger("caustic_photon_count", 10000);
            caustic_rad_estimation_count = props.getInteger("caustic_rad_estimation_count", 100);
        }

        Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const {
            /* Find the surface x that is visible in the requested direction */
            Intersection x, y;
            if(!scene->rayIntersect(ray, x))
                return Color3f(0.0f);

            /* initiation */
            Vector3f x_wi = -ray.d, x_wo;
            Color3f rad = Color3f(0.0f), throughput = Color3f(1.0f);
            BSDFQueryRecord x_bRec = BSDFQueryRecord(x.toLocal(x_wi));
            const BSDF *x_BSDF = x.mesh->getBSDF();

            /* collect light sources */
            std::vector<Mesh *> m_light_src;
            for (Mesh *mesh:scene->getMeshes()) {
                if (mesh->isEmitter())
                    m_light_src.push_back(mesh);
            }

            /* pass through all the specular surface */
            {
                while (!x_BSDF->isDiffuse()) {
                    x_BSDF->sample(x_bRec, sampler->next2D());
                    if (!scene->rayIntersect(Ray3f(x.p, -(x_wi = -x.toWorld(x_bRec.wo))), y))
                        return Color3f(0.0f);
                    x = y;
                    x_bRec.wi = x.toLocal(x_wi);
                    x_BSDF = x.mesh->getBSDF();
                }

                x_bRec.measure = ESolidAngle;

                /* le */
                if (x.mesh->isEmitter()) {
                    if (x.shFrame.n.dot(x_wi) > 0) {
                        return x.mesh->getEmitter()->getEnergy();
                    } else {
                        return Color3f(0.0f);
                    }
                }
            }


            /*========================================= Begin ==========================================*/
            /* direct illumination from light sources */
            for(Mesh *mesh:m_light_src){
                Point3f y_p; Normal3f y_n;
                mesh->getEmitter()->samplePosition(sampler->next1D(), sampler->next2D(), y_p, y_n);

                /* G(x<->y) */
                x_wo = y_p - x.p;
                float distance = x_wo.norm();
                x_wo.normalize();

                if (scene->rayIntersect(Ray3f(x.p, x_wo), y)
                    && std::abs(distance - y.t) < std::pow(10, -4)) {
                    /*G = <x_n, wo>*<y_n, -wo> /r^2 */
                    float G;
                    G = x.shFrame.n.dot(x_wo);       if (G < 0) continue;
                    G *= y_n.dot(-x_wo);             if (G < 0) continue;
                    G /= (y.t * y.t);

                    /* L_direct = Le * brdf(x, wi, wo) * G(x<->y)  */
                    x_bRec.wo = x.toLocal(x_wo);

                    rad +=  throughput * mesh->getEmitter()->getEnergy() * x_BSDF->eval(x_bRec) * G
                            / mesh->getEmitter()->pdf() ;
                }
            }

            /* caustic effect */
            rad += throughput * estimateIrradiance(x, x_wi, EPhotonMap::ECausticPhotonMap);

            /* next event */
            {
                throughput *= x_BSDF->sample(x_bRec, sampler->next2D());

                if (!scene->rayIntersect(Ray3f(x.p, -(x_wi = -x.toWorld(x_bRec.wo))), y)) {
                    return rad;
                }
                x = y;
                x_bRec.wi = x.toLocal(x_wi);
                x_BSDF = x.mesh->getBSDF();

                bool occlued_by_specular_object = false;
                while (!x_BSDF->isDiffuse()) {
                    occlued_by_specular_object = true;
                    x_BSDF->sample(x_bRec, sampler->next2D());
                    if (!scene->rayIntersect(Ray3f(x.p, -(x_wi = -x.toWorld(x_bRec.wo))), y))
                        return Color3f(0.0f);
                    x = y;
                    x_bRec.wi = x.toLocal(x_wi);
                    x_BSDF = x.mesh->getBSDF();
                }
                x_bRec.measure = ESolidAngle;

                /* le */
                if (occlued_by_specular_object && x.mesh->isEmitter() && x.shFrame.n.dot(x_wi) > 0) {
                    rad += throughput * x.mesh->getEmitter()->getEnergy();
                }
            }

            /*======================================== End ===========================================*/

            /* use RADIANCE ESTIMATION for indirect illumination */
            rad += throughput * estimateIrradiance(x, x_wi, EPhotonMap::EGlobalPhotonMap);

            return rad;
        }

        Color3f estimateIrradiance(Intersection position, Vector3f wi, EPhotonMap map) const {
            Intersection x = position;
            Point3f pos = x.p;
            const float query_pt[3] = {pos.x(), pos.y(), pos.z()};

            float radius_2, search_radius;
            size_t estimateCount, n_numPhotonShot;
            const my_kd_tree_t *m_index = nullptr;
            const PointCloud   *photons = nullptr;

            if(map == EPhotonMap::EGlobalPhotonMap) {
                estimateCount = rad_estimation_count;
                n_numPhotonShot = max_photon_count;
                search_radius = rad_estimation_radius;
                m_index = &index;
                photons = &cloud;
            } else {
                estimateCount = caustic_rad_estimation_count;
                n_numPhotonShot = caustic_max_photon_count;
                search_radius = rad_estimation_radius;
                m_index = &caustic_index;
                photons = &caustic_cloud;
            }

            size_t num_results = estimateCount;

            /// for N closet points search
            std::vector<std::pair<size_t, float>> ret_matches;
            nanoflann::SearchParams params;
            params.sorted = true;

            num_results = m_index->radiusSearch(&query_pt[0], search_radius, ret_matches, params);

            if(num_results > estimateCount) {
                num_results = estimateCount;
                radius_2 = ret_matches[num_results-1].second * ret_matches[num_results-1].second;
            } else {
                radius_2 = search_radius * search_radius;
            }

            BSDFQueryRecord x_bRec = BSDFQueryRecord(x.shFrame.toLocal(wi));
            const BSDF *x_BSDF = x.mesh->getBSDF();
            Color3f accflux = Color3f(0.0f);

            if(num_results > 0) {

                float invRadius_2 = 1/radius_2;

                for (size_t i = 0; i < num_results; i++) {

                    size_t idx = ret_matches[i].first;
                    PointCloud::Point photon = photons->pts[idx];
                    Vector3f wi = photon.wi;
                    Color3f power = photon.power;
                    x_bRec.wo = x.toLocal(wi);
                    x_bRec.measure = ESolidAngle;

                    float sqredDist = ret_matches[i].second * ret_matches[i].second;
                    float sqrTerm = 1 - sqredDist* invRadius_2;
                    accflux += power * (sqrTerm * sqrTerm) * x_BSDF->eval(x_bRec);
                }
                // 3 is for regulation due to the weight method
                float m_scale = 1 / (float)n_numPhotonShot;
                accflux *= m_scale * 3 * M_1_PI * invRadius_2;
            }

            return accflux;
        }

        std::string toString() const {
            return tfm::format(
                    "PhotonMappingIntegrator[\n"
                    "  GlobalPhotonMap:\n photon count = %i,\n"
                    "  radiance estimation count = %i,\n"
                    "  radiance estimation radius = %f,\n"
                    "  CausticPhotonMap:\n photon count = %i,\n"
                    "  radiance estimation count = %i,\n"
                    "  radiance estimation radius = %f,\n"
                    "]",
                    max_photon_count,
                    rad_estimation_count,
                    rad_estimation_radius,
                    caustic_max_photon_count,
                    caustic_rad_estimation_count,
                    rad_estimation_radius
            );
        }

    private:
        int max_photon_count, caustic_max_photon_count;
        int rad_estimation_count, caustic_rad_estimation_count;
        float rad_estimation_radius;


    };

    NORI_REGISTER_CLASS(PhotonMappingIntegrator, "photon_mapping");
NORI_NAMESPACE_END
