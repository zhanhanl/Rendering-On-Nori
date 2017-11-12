//
// Created by baconleung on 9/21/17.
//

#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/sampler.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>

#include <nori/nanoflann.hpp>
#include <ctime>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace nanoflann;

NORI_NAMESPACE_BEGIN

    class VCMIntegrator : public Integrator {
    public:

        enum EVertexMethod{
            EVertexConnection = 0,
            EVertexMerging
        };

        /* PM */
        struct PointCloud {
            struct Point
            {
                float  x,y,z;
                Color3f power;
                Vector3f wi;
                Intersection its;
                float pdf, geo;
                int order_on_the_path;
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

        PointCloud m_photons;
        my_kd_tree_t m_photonMap; //(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );

        void preprocess(const Scene *scene) {
            // Randomize Seed
            std::cout<<"building the photon map..."<<std::endl;
            std::cout<<"generate "<<max_photon_count << " photons for Global Photon Map"<<std::endl;

            // clone class Sampler
            std::unique_ptr<Sampler> sampler(scene->getSampler()->clone());

            // collect light sources
            // && sum up the area of all light sources
            std::vector<Mesh *> m_light_src;
            float totalArea = 0;
            for(Mesh *mesh:scene->getMeshes()) {
                if(mesh->isEmitter()) {
                    m_light_src.push_back(mesh);
                    totalArea += mesh->getEmitter()->getArea();
                }
            }

            // build the Photon Map
            m_photons.pts.clear();
            for(Mesh *mesh:m_light_src){
                Emitter *emitter = mesh->getEmitter();
                // distribute the amount of photons according to the light source's area
                int photon_count = max_photon_count * emitter->getArea() / totalArea;
                int n_emitted = 0;
                float invPPickLight = totalArea / emitter->getArea();

                while(n_emitted++ < photon_count){
                    Intersection x,y;
                    Point3f  y0_p;
                    Normal3f y0_n;

                    // sample a point on the light source
                    emitter->samplePosition(sampler->next1D(), sampler->next2D(), y0_p, y0_n);
                    x.p = y0_p;
                    x.mesh = mesh;
                    x.shFrame = Frame(y0_n);

                    // power = pi * A * Le
                    Color3f power = emitter->getEnergy() * emitter->getArea() * M_PI * invPPickLight;
                    int  current_idx        = 0;

                    // store photon on the light source
                    addPhotonToPhotonMap(x, (x.p + x.shFrame.n), power, 1, emitter->pdf(), current_idx++);

                    // photon tracing
                    int depth     = 5,    path_len = 0;
                    float survivalRate = m_survivalRate, probSurvival = 1; /* survival rate */
                    Vector3f wi   = x.shFrame.n;
                    Color3f throughput = Color3f(1.0f);
                    const BSDF *bsdf        = nullptr;
                    BSDFQueryRecord bRec    = BSDFQueryRecord(Vector3f(0,0,1));  bRec.measure = ESolidAngle;
                    bool sampledFromEmitter = true;

                    while(true){
                        bRec.wi = x.toLocal(wi);
                        bsdf = x.mesh->getBSDF();

                        // Fr(x0 -> x1) = 1
                        if(sampledFromEmitter){
                            bsdf->sample(bRec, sampler->next2D());
                            sampledFromEmitter = false;
                        }else {
                            throughput *= bsdf->sample(bRec, sampler->next2D());
                        }

                        // if there is no intersection, break the loop
                        if (!scene->rayIntersect(Ray3f(x.p, -(wi = -x.toWorld(bRec.wo))), y)) break;

                        float cosThetaX = std::abs(Frame::cosTheta(bRec.wo));
                        float cosThetaY = y.shFrame.n.dot(wi);
                        // cosThetaY can be <0 only when the hitted surface is specular, not diffuse.
                        if(cosThetaY < 0 && y.mesh->getBSDF()->isDiffuse()) break;

                        // add photon to PhotonMap( hasn't built the photonMap yet )
                        float pdf=0, geo = cosThetaX * std::abs(cosThetaY) / (y.t * y.t);
                        if(bsdf->isDiffuse()) {
                            pdf = bsdf->pdf(bRec) * probSurvival / cosThetaX;
                        } else {
                            pdf = probSurvival / cosThetaX;
                        }

                        addPhotonToPhotonMap(y, wi, power*throughput, geo, pdf, current_idx++);

                        // update
                        x = y;
                        path_len++;

                        // depth and the Russian Roulette
                        if(depth-- > 0) probSurvival = 1;
                        else if(sampler->next1D() > survivalRate || path_len > max_path_len) break;
                        else probSurvival = survivalRate;
                        throughput /= probSurvival;
                    }
                }
            }
            // build photonMap(KD-TREE)
            m_photonMap.buildIndex();

            cout<<"photons' size"<<m_photons.pts.size()<<endl;
            std::cout<<"done\n"<<std::endl;
        }

        inline void addPhotonToPhotonMap(Intersection x, Vector3f wi, Color3f power, float geo, float pdf, int order_on_path){
            PointCloud::Point photon;
            photon.x = x.p.x();
            photon.y = x.p.y();
            photon.z = x.p.z();
            photon.wi = wi;
            photon.power = power;
            photon.its = x;
            photon.order_on_the_path = order_on_path;
            photon.geo = geo;
            photon.pdf = pdf;
            m_photons.pts.push_back(photon);
        }


        /* BDPT */

        inline float PProjectedSolidAngle(const Scene *scene, Point3f xi, Intersection x, Point3f xo) const {
            BSDFQueryRecord bRec = BSDFQueryRecord(x.shFrame.toLocal((xi-x.p).normalized()));
            Vector3f wo =(xo-x.p).normalized();
            bRec.wo = x.shFrame.toLocal(wo);
            bRec.measure = ESolidAngle;

            if(x.mesh->getBSDF()->isDiffuse())
                return x.mesh->getBSDF()->pdf(bRec)/x.shFrame.n.dot(wo);
            else
                return 1 / x.shFrame.n.dot(wo);
        }

        inline Color3f Fr(const Scene *scene, Point3f xi, Intersection x, Point3f xo) const {
            BSDFQueryRecord bRec = BSDFQueryRecord(x.shFrame.toLocal((xi-x.p).normalized()));
            bRec.wo = x.shFrame.toLocal((xo-x.p).normalized());
            bRec.measure = ESolidAngle;
            return x.mesh->getBSDF()->eval(bRec);
        }

        inline float G(const Scene *scene, Intersection x, Intersection y) const {
            Intersection t;
            Vector3f wo = y.p - x.p;
            float dis = wo.norm();
            wo.normalize();

            if(!scene->rayIntersect(Ray3f(x.p, wo), t)
               || std::abs(t.t - dis) > 0.0001)
                return 0.0f;

            float g  = x.shFrame.n.dot(wo);                   if(g<0) return 0.0f;
            g       *= y.shFrame.n.dot(-wo);                  if(g<0) return 0.0f;
            g       /= dis*dis;

            return g;
        }

        bool generatePath(const Scene *scene, Sampler *sampler, Intersection x, Vector3f wi,
                          std::vector<Intersection> &its, std::vector<float> &pdf, std::vector<float> &g) const {

            int   depth = 5,   path_len = 0;
            float SRVL = m_survivalRate, pSRVL, mpdf=1, mg=1;

            Intersection   y;
            BSDFQueryRecord bRec = BSDFQueryRecord(x.toLocal(wi));
            bRec.measure     = ESolidAngle;
            const BSDF *bsdf = nullptr;

            while (true) {
                its.push_back(x);
                pdf.push_back(mpdf);
                g.push_back(mg);

                // depth and the Russian roulette
                if (depth-- > 0) pSRVL = 1;
                else if (sampler->next1D() > SRVL || path_len > max_path_len) return true;
                else pSRVL = SRVL;

                bRec.wi = x.toLocal(wi);
                bsdf = x.mesh->getBSDF();
                bsdf->sample(bRec, sampler->next2D());

                if (!scene->rayIntersect(Ray3f(x.p, -(wi = -x.toWorld(bRec.wo))), y)) {
                    return false;
                }

                float cosThetaX = std::abs(Frame::cosTheta(bRec.wo));
                float cosThetaY = y.shFrame.n.dot(wi);

                if(cosThetaY < 0 && y.mesh->getBSDF()->isDiffuse()) return false;

                if(!bsdf->isDiffuse()) {
                    mpdf = pSRVL / cosThetaX;
                }
                else {
                    mpdf = bsdf->pdf(bRec) * pSRVL / cosThetaX;
                }

                mg = cosThetaX * std::abs(cosThetaY) /(y.t * y.t);
                x = y;
                path_len ++;
            }  // while(true){...}
        }




        /* Integrator */

        VCMIntegrator(const PropertyList &props)
                : m_photonMap(3 /* dim */, m_photons, KDTreeSingleIndexAdaptorParams(10 /* max leaf */)) {
            max_photon_count = props.getInteger("photon_count", 10000);
            rad_est_count    = props.getInteger("rad_estimation_count", 100);
            rad_est_radius   = props.getFloat("rad_estimation_radius", 0.05);
            max_path_len     = props.getInteger("max_path_len", 20);
            gau_alpha        = props.getFloat("alpha", 1.818);
            gau_beta         = props.getFloat("beta", 1.953);
            m_survivalRate   = 0.9f;
            len_of_eye_path  = props.getInteger("t", 1);
            m_scale          = 1 / (float)max_photon_count;
        }

        Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const{
            // find a visible point in the given direction
            Intersection x;
            if (!scene->rayIntersect(ray, x)) return Color3f(0.0f);

            // if it hits a light source, return 0 if it is the back
            // of the light, or the direct radiance of the light source
            if (x.mesh->isEmitter()) return x.shFrame.n.dot(-ray.d) > 0 ? x.mesh->getEmitter()->getEnergy() : Color3f(0.0f);

            Color3f  rad   = Color3f(0.0f);
            Vector3f wi    = -ray.d;

            // collect all light sources
            Emitter *m_emitter = nullptr;
            Mesh    *m_light   = nullptr;
            std::vector<Mesh *> m_lights;
            for (Mesh *mesh:scene->getMeshes()) if(mesh->isEmitter()) m_lights.push_back(mesh);


            /* ---- bdpt : generate paths ---- */
            // EyePath vs LightPath
            //  +--------------------------------------------+
            //  |index type        |  i-th  |  # of vertices |
            //  |m_alphaEyePath    |        |      X         |
            //  |m_jointPdfEyePath |        |      X         |
            //  |m_itsEyePath      |   X    |                |
            //  |m_pdfEyePath      |   X    |                |
            //  |m_geoEyePath      |   X    |                |
            //  +--------------------------------------------+
            //  ***  m_*LightPath is the same as m_*EyePath

            std::vector<Intersection> m_itsEyePath,      m_itsLightPath;
            std::vector<float>        m_pdfEyePath,      m_pdfLightPath;
            std::vector<float>        m_geoEyePath,      m_geoLightPath;
            std::vector<Color3f>      m_alphaEyePath,    m_alphaLightPath;
            std::vector<float>        m_jointPdfEyePath, m_jointPdfLightPath;

            /* ==================================================================== */
            /*                     Generate Eye Path                                */
            /* ==================================================================== */
            {
                // use i-th to index m_itsEyePath, m_pdfEyePath, m_geoEyePath
                m_itsEyePath.clear();  m_pdfEyePath.clear();  m_geoEyePath.clear();
                generatePath(scene, sampler, x, wi, m_itsEyePath, m_pdfEyePath, m_geoEyePath);

                // compute the Alpha
                // use # of vertices to index m_alphaEyePath
                m_alphaEyePath.clear();
                m_alphaEyePath.push_back(Color3f(1.0f));  // alpha_eye[0] = 1
                m_alphaEyePath.push_back(Color3f(1.0f));  // alpha_eye[1] = 1
                // alpha_eye[2]
                if(m_itsEyePath.size() > 1 ) {
                    if(m_itsEyePath[0].mesh->getBSDF()->isDiffuse())
                        m_alphaEyePath.push_back(Fr(scene, ray.o, m_itsEyePath[0], m_itsEyePath[1].p)
                                                 / m_pdfEyePath[1] * m_alphaEyePath[1]);   // diffuse  surface
                    else
                        m_alphaEyePath.push_back(1 * m_alphaEyePath[1]);                   // specular surface
                }
                // alpha_eye[i]
                for(unsigned i = 3; i <= m_itsEyePath.size(); i++) {
                    // alpha_i = BSDF / pdf * alpha_i-1
                    if(m_itsEyePath[i-2].mesh->getBSDF()->isDiffuse())
                        m_alphaEyePath.push_back(Fr(scene, m_itsEyePath[i-3].p, m_itsEyePath[i-2], m_itsEyePath[i-1].p)
                                                 / m_pdfEyePath[i-1] * m_alphaEyePath[i-1]); // diffuse  surface
                    else
                        m_alphaEyePath.push_back(1 * m_alphaEyePath[i-1]);                   // specualr surface
                }
            }

            /* ==================================================================== */
            /*                     Generate Light Path                              */
            /* ==================================================================== */
            {
                /* randomly pick one light source */
                int indexPickLightSource = std::floor(sampler->next1D() * m_lights.size());
                float probPickLightSource = 1 / m_lights.size();
                m_light   = m_lights[indexPickLightSource];
                m_emitter = m_light->getEmitter();

                Intersection y;
                Point3f positionY0;  Normal3f normalY0;
                /* sample a position on the light source */
                m_emitter->samplePosition(sampler->next1D(), sampler->next2D(), positionY0, normalY0);
                modifyIntersection(y, positionY0, normalY0, m_light);

                generatePath(scene, sampler, y, normalY0, m_itsLightPath, m_pdfLightPath, m_geoLightPath);

                /* reset some parameter of light path */
                /* P(y0) = (1/A) * Ppicklight */
                m_pdfLightPath[0] = m_emitter->pdf() * probPickLightSource;

                /* compute ALPHA */
                /* alpha0 = 1 */
                m_alphaLightPath.push_back(Color3f(1.0f));
                /* alpha1 = Le / (PA * Ppicklight) = Le * A * numLights */
                m_alphaLightPath.push_back(m_emitter->getEnergy() / m_pdfLightPath[0]);
                /* alpha2 */
                if(m_itsLightPath.size()>= 2) {
                    /* diffuse , Fr(y0 -> y1) = 1 */
                    if(m_itsLightPath[0].mesh->getBSDF()->isDiffuse())
                        m_alphaLightPath.push_back( m_alphaLightPath[1] / m_pdfLightPath[1] );
                    else    /* specular */
                        m_alphaLightPath.push_back( m_alphaLightPath[1]);
                }
                for (unsigned i = 3; i <= m_itsLightPath.size(); i++) {
                    if(m_itsLightPath[i-2].mesh->getBSDF()->isDiffuse()) {
                        Color3f bsdfFr = Fr(scene, m_itsLightPath[i - 3].p, m_itsLightPath[i - 2],
                                            m_itsLightPath[i - 1].p);
                        m_alphaLightPath.push_back(m_alphaLightPath[i-1] * bsdfFr / m_pdfLightPath[i - 1]);
                    }
                    else {
                        m_alphaLightPath.push_back( m_alphaLightPath[i-1]);
                    }
                }
            }


            /* ==================================================================== */
            /*                     Vertex Connection and Merging                    */
            /* ==================================================================== */
            {
                int     eyeVerticesSize   = m_itsEyePath.size();
                const float radius_sqr_fixed = rad_est_radius * rad_est_radius;
                float   radius_sqr = radius_sqr_fixed;
                eyeVerticesSize = eyeVerticesSize > 10 ? 10 : eyeVerticesSize;
                Color3f FrLightPath, FrEyePath;

                /* TODO: pass through all the specualr surface */

                for(int t = 1; t <= eyeVerticesSize; t++) {

                    /* ==================================================================== */
                    /*                     Vertex Merging                                   */
                    /* ==================================================================== */
                    if(m_itsEyePath[t-1].mesh->getBSDF()->isDiffuse()) {
                        // "x" is end point of the eye path
                        Intersection    x       = m_itsEyePath[t - 1];
                        const BSDF     *x_BSDF  = x.mesh->getBSDF();
                        BSDFQueryRecord x_bRec  = BSDFQueryRecord(Vector3f(0,0,1));  x_bRec.measure = ESolidAngle;

                        // incident radiance
                        if(t == 1) {
                            x_bRec.wi = x.toLocal((ray.o - m_itsEyePath[t-1].p).normalized());
                        } else {
                            x_bRec.wi = x.toLocal((m_itsEyePath[t - 2].p - m_itsEyePath[t - 1].p).normalized());
                        }

                        // for N closest points search, search by radius
                        const float query_pt[3]   = {x.p.x(), x.p.y(), x.p.z()};
                        float search_radius = rad_est_radius;
                        std::vector<std::pair<size_t, float>> ret_matches;
                        nanoflann::SearchParams params;  params.sorted = true;

                        size_t numRadianceEstimate = rad_est_count;
                        size_t num_results = m_photonMap.radiusSearch(&query_pt[0], search_radius, ret_matches, params);

                        if(num_results > numRadianceEstimate) {
                            num_results = numRadianceEstimate;
                            radius_sqr = ret_matches[num_results-1].second * ret_matches[num_results-1].second;
                        } else {
                            radius_sqr = search_radius * search_radius;
                        };

                        if (num_results > 0) {
                            Color3f accumulatedFlux = Color3f(0.0f);
                            float invRadiusSquard = 1 / radius_sqr;

                            // iterate over the returned photons
                            for (size_t i = 0; i < num_results; i++) {
                                size_t idx       = ret_matches[i].first;
                                PointCloud::Point photon = m_photons.pts[idx];

                                // retrive itersection, pdf and geometry term of the light path
                                std::vector<Intersection> itsLightPath;
                                std::vector<float> pdfLightPath, geoLightPath;
                                int order = photon.order_on_the_path;
                                // indexLightSource = indexCurrentPhoton - orderCurrentPhotonOnPath
                                idx = idx - order;
                                for(int j = 0; j <= order; j++, idx++) {
                                    itsLightPath.push_back(m_photons.pts[idx].its);
                                    pdfLightPath.push_back(m_photons.pts[idx].pdf);
                                    geoLightPath.push_back(m_photons.pts[idx].geo);
                                }

                                float weight = MIS_VCM(scene, ray, itsLightPath.size(), t, radius_sqr, EVertexMerging,
                                                       itsLightPath, m_itsEyePath,
                                                       pdfLightPath, m_pdfEyePath,
                                                       geoLightPath, m_geoEyePath);

                                x_bRec.wo    = x.toLocal(photon.wi);

                                float sqrDist = ret_matches[i].second * ret_matches[i].second;
                                float sqrTerm = 1 - sqrDist * invRadiusSquard;
                                weight *= sqrTerm * sqrTerm;
                                accumulatedFlux += x_BSDF->eval(x_bRec) * photon.power * weight;
                            }

                            rad += m_alphaEyePath[t] * m_scale * accumulatedFlux * M_1_PI * invRadiusSquard * 3;
                        }
                    }

                    /* ==================================================================== */
                    /*                     Vertex Connection                                */
                    /* ==================================================================== */
                    /* vertex of eye path is on the light source*/
                    if (m_itsEyePath[t - 1].mesh == m_light) {
                        float weight = MIS_VCM(scene, ray, 0, t, radius_sqr, EVertexConnection,
                                               m_itsLightPath, m_itsEyePath,
                                               m_pdfLightPath, m_pdfEyePath,
                                               m_geoLightPath, m_geoEyePath);
                        rad += m_alphaEyePath[t] * m_emitter->getEnergy() * weight;
                    }

                    for(int s=1; s<=m_itsLightPath.size(); s++) {
                        FrLightPath = s==1? Color3f(1.0f) : Fr(scene, m_itsLightPath[s-2].p, m_itsLightPath[s-1], m_itsEyePath[t-1].p);
                        FrEyePath   = t==1? Fr(scene, ray.o, m_itsEyePath[t-1], m_itsLightPath[s-1].p)
                                          : Fr(scene, m_itsEyePath[t-2].p, m_itsEyePath[t-1], m_itsLightPath[s-1].p);
                        float GTerm = G(scene, m_itsEyePath[t-1], m_itsLightPath[s-1]);
                        if(GTerm == 0) continue;
                        float weight = MIS_VCM(scene, ray, s, t, radius_sqr, EVertexConnection,
                                               m_itsLightPath, m_itsEyePath,
                                               m_pdfLightPath, m_pdfEyePath,
                                               m_geoLightPath, m_geoEyePath);
                        rad += m_alphaLightPath[s] * FrLightPath * GTerm * FrEyePath * m_alphaEyePath[t] * weight;

                    }

                }   // iterate the eye path
            }
            return rad;
        }

        inline void modifyIntersection(Intersection &x, Point3f position, Normal3f normal, Mesh *mesh) const {
            x.p = position;
            x.mesh = mesh;
            x.shFrame = Frame(normal);
        }


        /*
         * Pvc,s,t(_x_) = Ps(_y_)Pt(_z_)
         * Pvm,s,t(_x_) = Ps(_y_)Pt(_z_) *(pi*r*r)
         *
         * Path for Pvc,s,t(_x_):
         *  Y0 -->...--> Ys-2 --> Ys-1 ------ Zt-1 <--...<-- Z0
         *
         * Path for Pvm,s,t(_x_):
         *  Y0 -->...--> Ys-2 --> Ys-1_Zt-1 <--...<-- Z0
         * */
        float MIS_VCM(const Scene *scene, Ray3f ray, int _s, int _t,  float radius_sqr, EVertexMethod method,
                      std::vector<Intersection> &itsLightPath, std::vector<Intersection> &itsEyePath,
                      std::vector<float> &pdfLightPath,        std::vector<float> &pdfEyePath,
                      std::vector<float> &geoLightPath,        std::vector<float> &geoEyePath) const {
            int numVC = 1, numVM = max_photon_count;
            float probVC, probVM, probCurrent, accumulatedProbVC=0, accumulatedProbVM=0;
            float discreteLightPathPDF, discreteEyePathPDF;
            float areaRadianceEstimate = M_PI * radius_sqr;
            int s, t;
            Intersection x, y;
            Point3f xi;

            if(method == EVertexMethod::EVertexMerging){ if(_s<2 || _t<1) return 0; }

            /* ==================================================================== */
            /*                     Vertex Connection Weighting                      */
            /* ==================================================================== */
            if(method == EVertexMethod::EVertexConnection) {
                probVC = 1;
                s = _s;
            } else {
                probVC = pdfLightPath[_s-1] * geoLightPath[_s-1] * areaRadianceEstimate;
                probVC = 1 / probVC;
                s = _s-1;
            }
            t = _t;
            if(s>=1 && itsLightPath[s-1].mesh->getBSDF()->isDiffuse())
                accumulatedProbVC += probVC;

            /* sample path from eye to light */
            probCurrent = probVC;
            if(s >= 1) {
                /* assign end points:
                 * x on eye path, y on light path */
                xi = t == 1? ray.o: itsEyePath[t-2].p;
                x  = itsEyePath[t-1];
                y  = itsLightPath[s-1];

                for(int ss = s; ss>=2; ss--) {
                    discreteLightPathPDF = pdfLightPath[ss-1] * geoLightPath[ss-1];
                    if(x.mesh->getBSDF()->isDiffuse()) {
                        discreteEyePathPDF = PProjectedSolidAngle(scene, xi, x, y.p) * G(scene, x, y);
                    } else {
                        discreteEyePathPDF = G(scene, x, y) / x.shFrame.n.dot((y.p - x.p).normalized());
                    }

                    probCurrent *= discreteEyePathPDF / discreteLightPathPDF;
                    xi = x.p;  x = y;  y = itsLightPath[ss-2];

                    if(x.mesh->getBSDF()->isDiffuse() && y.mesh->getBSDF()->isDiffuse())
                        accumulatedProbVC += probCurrent;
                }

                /* for case where s=1, end point of eye path is on the light source,
                 * and assume that light source can't be specular */
                discreteLightPathPDF = pdfLightPath[0];
                discreteEyePathPDF   = PProjectedSolidAngle(scene, xi, x, y.p) * G(scene, x, y);
                accumulatedProbVC += discreteEyePathPDF / discreteLightPathPDF;
            }

            /* sample path from light to eye */
            probCurrent = probVC;
            if(s == 0) {
                discreteLightPathPDF = pdfLightPath[0];
                discreteEyePathPDF   = pdfEyePath[t-1] * geoEyePath[t-1];
                probCurrent *= discreteLightPathPDF / discreteEyePathPDF;
                /* if an eye path can hit the light source, it must have at least two
                 * vertices, one on the light source, the other on the surface,
                 * and assume that light source can't be specular */
                if(itsEyePath[t-2].mesh->getBSDF()->isDiffuse())
                    accumulatedProbVC += probCurrent;

                /* assign end points:
                 * x on light path, y on eye path */
                xi = itsEyePath[t-1].p + itsEyePath[t-1].shFrame.n;
                x  = itsEyePath[t-1];
                y  = itsEyePath[t-2];
                t--;
            } else {
                xi = s==1? (itsLightPath[0].p+itsLightPath[0].shFrame.n) : itsLightPath[s-2].p;
                x  = itsLightPath[s-1];
                y  = itsEyePath[t-1];
            }

            for(int tt=t; tt>=2; tt--) {
                discreteEyePathPDF = pdfEyePath[tt-1] * geoEyePath[tt-1];
                if(x.mesh->getBSDF()->isDiffuse()) {
                    discreteLightPathPDF = PProjectedSolidAngle(scene, xi, x, y.p) * G(scene, x, y);
                } else {
                    discreteLightPathPDF = G(scene, x, y) / x.shFrame.n.dot((y.p-x.p).normalized());
                }

                probCurrent *= discreteLightPathPDF / discreteEyePathPDF;
                xi = x.p;  x = y;  y = itsEyePath[tt-2];

                if(x.mesh->getBSDF()->isDiffuse() && y.mesh->getBSDF()->isDiffuse())
                    accumulatedProbVC += probCurrent;
            }



            /* ==================================================================== */
            /*                     Vertex Merging Weighting                         */
            /* ==================================================================== */
            vector<float> pdfLightPathVM, geoLightPathVM;
            vector<Intersection> itsLightPathVM;

            /* change the varible to xxxLightPathVM
             * copy vector by assign function*/
            itsLightPathVM.assign(itsLightPath.begin(), itsLightPath.end());
            pdfLightPathVM.assign(pdfLightPath.begin(), pdfLightPath.end());
            geoLightPathVM.assign(geoLightPath.begin(), geoLightPath.end());


            if(method == EVertexMethod::EVertexMerging) {
                probVM = 1;
                s = _s;
                t = _t;
            } else {
                /* x : end point of light path
                 * y : end point of eye path */
                if(_s == 0) {
                    if(_t == 1) { cout<<"Error: with s=0 and t=1."<<endl; return 0; }

                    /* generate a vertex on light source */
                    itsLightPathVM.push_back(itsEyePath[_t-1]);
                    /* TODO : times probPickLightSource*/
                    pdfLightPathVM.push_back(itsEyePath[_t-1].mesh->getEmitter()->pdf());
                    geoLightPathVM.push_back(1);

                    xi = itsEyePath[_t-1].p + itsEyePath[_t-1].shFrame.n;
                    x  = itsEyePath[_t-1];
                    y  = itsEyePath[_t-2];

                    t = _t-1;
                    s = 2;
                } else {
                    xi = _s==1? itsLightPath[_s-1].p + itsLightPath[_s-1].shFrame.n : itsLightPath[_s-2].p;
                    x  = itsLightPath[_s-1];
                    y  = itsEyePath[_t-1];

                    s = _s+1;
                    t = _t;
                }
                /* add on the extra point Xs* to the new light path */
                itsLightPathVM.push_back(y);
                pdfLightPathVM.push_back(PProjectedSolidAngle(scene, xi, x, y.p));
                geoLightPathVM.push_back(G(scene, x, y));

                probVM = pdfLightPathVM[s-1] * geoLightPathVM[s-1] * areaRadianceEstimate;
            }
            if(itsLightPathVM[s-1].mesh->getBSDF()->isDiffuse())
                accumulatedProbVM += probVM;

            /* sample path from eye to light
             * if s=2, no need to calculate the probability */
            probCurrent = probVM;
            if(s >= 3) {
                /* assign end points:
                 * x on eye path, y on light path */
                xi = t==1? ray.o : itsEyePath[t-2].p;
                x  = itsEyePath[t-1];
                y  = itsLightPathVM[s-2];

                /* visibility test for the new extra point Xs*
                 * 1. point Xs and Xs* are blocked
                 * 2. the pdf will be 0 or NaN when the photon arrives on one side
                 *    of the corner and the eye path locate on the other side of
                 *    the corner.
                 * */
                discreteEyePathPDF = PProjectedSolidAngle(scene, xi, x, y.p) * G(scene, x, y);
                if(discreteEyePathPDF == 0 || discreteEyePathPDF != discreteEyePathPDF) {
                    accumulatedProbVM = 0;
                    probVM = 0;
                    if(method == EVertexMerging) return 0;
                } else {
                    for(int ss=s; ss >=3; ss--) {
                        discreteLightPathPDF = pdfLightPathVM[ss-1] * geoLightPathVM[ss-1];
                        if(x.mesh->getBSDF()->isDiffuse()) {
                            discreteEyePathPDF = PProjectedSolidAngle(scene, xi, x, y.p) * G(scene, x, y);
                        } else {
                            discreteEyePathPDF = changeMeasureFromSolidAngleToArea(x, y);
                        }

                        probCurrent *= discreteEyePathPDF / discreteLightPathPDF;
                        xi = x.p;  x = y;   y = itsLightPathVM[ss-3];

                        if(x.mesh->getBSDF()->isDiffuse())
                            accumulatedProbVM += probCurrent;
                    }
                }
            }

            /* sample path from light to eye */
            probCurrent = probVM;
            if(t >= 2 && s >= 2 && probVM != 0) {
                /* assign end points */
                xi = itsLightPathVM[s-2].p;
                x  = itsEyePath[t-1];
                y  = itsEyePath[t-2];
                for(int tt=t; tt>=3; tt--) {
                    discreteEyePathPDF = pdfEyePath[tt-1] * geoEyePath[tt-1];
                    if(x.mesh->getBSDF()->isDiffuse()) {
                        discreteLightPathPDF = PProjectedSolidAngle(scene, xi, x, y.p) * G(scene, x, y);
                    } else {
                        discreteLightPathPDF = changeMeasureFromSolidAngleToArea(x, y);
                    }

                    probCurrent *= discreteLightPathPDF / discreteEyePathPDF;
                    xi = x.p;  x = y;  y = itsEyePath[tt-3];

                    if(x.mesh->getBSDF()->isDiffuse())
                        accumulatedProbVM += probCurrent;
                }
                /* when t=2, for Pvm,s+t-1,t */
                discreteEyePathPDF = pdfEyePath[1] * geoEyePath[1];
                if(x.mesh->getBSDF()->isDiffuse()) {
                    discreteLightPathPDF = PProjectedSolidAngle(scene, xi, x, y.p) * G(scene, x, y);
                } else {
                    discreteLightPathPDF = changeMeasureFromSolidAngleToArea(x, y);
                }
                if(y.mesh->getBSDF()->isDiffuse())
                    accumulatedProbVM += discreteLightPathPDF / discreteEyePathPDF;
            }

            float result;
            if(method == EVertexMethod::EVertexMerging) {
                result = numVC * accumulatedProbVC + numVM * accumulatedProbVM;
                result = numVM / result;
            }
            else {
                result = numVC * accumulatedProbVC + numVM * accumulatedProbVM;
                result = numVC / result;
            }

            if(std::isinf(result) || std::isnan(result)) result = 0;
            return result;

        }


        // result = |cosThetaY| / dist^2
        inline float changeMeasureFromSolidAngleToArea(Intersection x, Intersection y) const {
            Vector3f YtoX = x.p - y.p;
            float invDistSqr = 1 / YtoX.squaredNorm();
            float cosThetaY = std::abs(y.shFrame.n.dot(YtoX.normalized()));
            return cosThetaY * invDistSqr;
        }


        std::string toString() const {
            return tfm::format(
              "VCMIntegrator[\n"
              " photon count = %i, \n"
              " radiance estimation radius = %f,\n"
              " alpha = %f, beta = %f\n"
              "]",
              max_photon_count,
              rad_est_radius,
              gau_alpha,
              gau_beta
            );
        }

    private:
        int   max_photon_count;
        int   max_path_len;
        float rad_est_radius, m_scale;
        int   rad_est_count;
        float gau_alpha, gau_beta;
        float m_survivalRate;
        int   len_of_eye_path;

    };

    NORI_REGISTER_CLASS(VCMIntegrator, "vcm");
NORI_NAMESPACE_END