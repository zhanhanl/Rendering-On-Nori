//
// Created by baconleung on 6/29/17.
//
#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/bsdf.h>
#include <nori/sampler.h>
#include <nori/emitter.h>

NORI_NAMESPACE_BEGIN
/* bi-directional path tracing */
    class BDPTIntegrator : public Integrator {
    public:
        BDPTIntegrator(const PropertyList &props) {
            /* do nothing here */
        }

        Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const {
            /* Find the surface x that is visible in the requested direction */
            Intersection x;
            if (!scene->rayIntersect(ray, x))
                return Color3f(0.0f);

            if (x.mesh->isEmitter())
                return x.shFrame.n.dot(-ray.d) > 0 ? x.mesh->getEmitter()->getEnergy() : Color3f(0.0f);

            /* collect light sources */
            std::vector<Mesh *> m_lights;
            for (Mesh *mesh:scene->getMeshes()) if (mesh->isEmitter()) m_lights.push_back(mesh);

            /* T: eye, importance ;
             * L: light */
            std::vector<Intersection> m_itsT, m_itsS;
            std::vector<Color3f> m_alphaT, m_alphaS;
            std::vector<float> m_pdfT, m_pdfS;
            std::vector<float> m_GT, m_GS;
            Color3f rad = Color3f(0.0f);
            Vector3f wi = -ray.d;

            /* generate IMPORTANCE path */
            {
                generatePath(scene, sampler, x, wi, m_itsT, m_pdfT, m_GT);

                /* compute: path sampling technique, alpha */

                /* alpha
                 * i vertices corresponds to m_alphaT[i]
                 * */
                m_alphaT.push_back(Color3f(1.0f)); /* alpha0^E = 1 */
                m_alphaT.push_back(Color3f(1.0f)); /* alpha1^E = 1 */
                /* i=2 */
                if(m_itsT.size()>1) { /* alpha2^E */
                    if (m_itsT[0].mesh->getBSDF()->isDiffuse())
                        /* diffuse */
                        m_alphaT.push_back(Fr(scene, ray.o, m_itsT[0], m_itsT[1].p) / m_pdfT[1] * m_alphaT[1]);
                    else
                        /* specular */
                        m_alphaT.push_back(1 * m_alphaT[1]);
                }
                for (unsigned i = 3; i <= m_itsT.size(); i++) {
                    if(m_itsT[i-2].mesh->getBSDF()->isDiffuse())
                        /* diffuse */
                        m_alphaT.push_back(Fr(scene, m_itsT[i-3].p, m_itsT[i-2], m_itsT[i-1].p) / m_pdfT[i - 1] * m_alphaT[i - 1]);
                    else
                        /* specular */
                        m_alphaT.push_back(1 * m_alphaT[i-1]);
                }
            }

            for (Mesh *mesh:m_lights) {

                m_pdfS.clear();
                m_GS.clear();
                m_itsS.clear();

                Point3f  y0_p;
                Normal3f y0_n;
                Emitter *emitter = mesh->getEmitter();
                emitter->samplePosition(sampler->next1D(), sampler->next2D(), y0_p, y0_n);

                /* generate LIGHT path */
                {
                    Intersection y;
                    y.p = y0_p;
                    y.mesh = mesh;
                    y.shFrame = Frame(y0_n);
                    generatePath(scene, sampler, y, y0_n, m_itsS, m_pdfS, m_GS);
                    m_pdfS[0] = emitter->pdf(); /* */
                    /* compute: path sampling technique, alpha */
                    /* alpha
                    * i vertices corresponds to m_alphaS[i]
                    * */
                    m_alphaS.push_back(Color3f(1.0f)); /* alpha0^L=1*/
                    m_alphaS.push_back(emitter->getEnergy() / emitter->pdf()); /* alpha1^L = Le/PA */
                    /* i=2 */
                    if(m_itsS.size()>1) {/* alpha2^L */
                        if(m_itsS[0].mesh->getBSDF()->isDiffuse())
                            /* diffuse */
                            m_alphaS.push_back( 1 / m_pdfS[1] * m_alphaS[1]);
                        else
                            /* specular */
                            m_alphaS.push_back( 1 * m_alphaS[1]);
                    }
                    for (unsigned i = 3; i <= m_itsS.size(); i++) {
                        if(m_itsS[i-2].mesh->getBSDF()->isDiffuse())
                            m_alphaS.push_back(Fr(scene, m_itsS[i-3].p, m_itsS[i-2], m_itsS[i-1].p) / m_pdfS[i - 1] * m_alphaS[i - 1]);
                        else
                            m_alphaS.push_back( 1 * m_alphaS[i - 1]);
                    }
                }


                /* bi-directional path tracing */
                {
                    Color3f FrS, FrT;
                    Vector3f wi, wo;

                    int L = m_itsS.size(), E = m_itsT.size();

                    // k := # of total vertices
                    for (int k = 2; k <= L + E; k++) {

                        /* s=0, t=k */
                        if(k<=E && m_itsT[k-1].mesh == mesh
                                && m_itsT[k-1].shFrame.n.dot((m_itsT[k-2].p - m_itsT[k-1].p).normalized())>0){
                            float pcur = 1, w = 1;
                            Point3f xi,xo;
                            Intersection x,y;

                            /* backward, in direction from light to eye */
                            pcur *= m_pdfS[0] / (m_pdfT[k-1] * m_GT[k-1]);
                            if(m_itsT[k-1].mesh->getBSDF()->isDiffuse()
                               && m_itsT[k-2].mesh->getBSDF()->isDiffuse()){
                                w+=pcur;
                            }

                            if(k>2) {
                                xi = (m_itsT[k - 1].p + m_itsT[k - 1].shFrame.n);
                                x = m_itsT[k - 1];
                                xo = m_itsT[k - 2].p;
                                y = m_itsT[k - 2];

                                for (int tt = k - 1; tt > 1; tt--) {
                                    if(!x.mesh->getBSDF()->isDiffuse()) {
                                        pcur *=  1 / x.shFrame.n.dot((xo - x.p).normalized()) * G(scene, x, y) / (m_pdfT[tt - 1] * m_GT[tt - 1]);
                                    }else{
                                        pcur *= PProjectedSolidAngle(scene, xi, x, xo) * G(scene, x, y) / (m_pdfT[tt - 1] * m_GT[tt - 1]);
                                    }

                                    if(m_itsT[tt-1].mesh->getBSDF()->isDiffuse()
                                       && m_itsT[tt-2].mesh->getBSDF()->isDiffuse()){
                                        w+=pcur;
                                    }

                                    /* update */
                                    xi = x.p;
                                    x = m_itsT[tt - 1];
                                    xo = m_itsT[tt - 2].p;
                                    y = m_itsT[tt - 2];
                                }
                            }

                            w = 1 / w;
                            if(w!=w) w=0;
                            rad += emitter->getEnergy() * m_alphaT[k] * w;

                        }


                        for(int t=1, s=k-t; t<=E && s>0; t++, s--){
                            if (s > L) continue;

                            /* light path [s-1] */
                            if (s == 1) {
                                FrS = Color3f(1.0f);
                            } else {
                                FrS = Fr(scene, m_itsS[s-2].p, m_itsS[s-1], m_itsT[t-1].p);
                            }

                            /* importance path [t-1] */
                            if (t == 1) {
                                FrT = Fr(scene, ray.o, m_itsT[t-1], m_itsS[s-1].p);
                            } else {
                                FrT = Fr(scene, m_itsT[t-2].p, m_itsT[t-1], m_itsS[s-1].p);
                            }

                            float g = G(scene, m_itsT[t-1], m_itsS[s-1]);

                            if (g!=0) {

                                float w = 1, pcur = 1;
                                /* weighting function */
                                {
                                    /* calculating the weight */
                                    Point3f xi,xo;
                                    Intersection x,y;
                                    /* forward, in direction from eye to light */
                                    /* i,k,t,s : # of vertices */
                                    if(t==1) xi = ray.o;
                                    else xi = m_itsT[t-2].p;

                                    x = m_itsT[t-1];
                                    xo = m_itsS[s-1].p;
                                    y = m_itsS[s-1];

                                    for(int ss=s ; ss>1; ss--){

                                        if(!x.mesh->getBSDF()->isDiffuse()) {
                                            pcur *=  G(scene, x, y) / x.shFrame.n.dot((xo - x.p).normalized())  / (m_pdfS[ss - 1] * m_GS[ss - 1]);
                                        }else{
                                            pcur *= PProjectedSolidAngle(scene, xi, x, xo) * G(scene, x, y) / (m_pdfS[ss - 1] * m_GS[ss - 1]);
                                        }

                                        if(m_itsS[ss-1].mesh->getBSDF()->isDiffuse()
                                           && m_itsS[ss-2].mesh->getBSDF()->isDiffuse()){
                                            w+=pcur;
                                        }

                                        /* update */
                                        xi = x.p;
                                        x = m_itsS[ss-1];
                                        xo = m_itsS[ss-2].p;
                                        y = m_itsS[ss-2];
                                    }

                                    /* s = 1 */
                                    pcur *= PProjectedSolidAngle(scene, xi, x, xo) * G(scene,x,y)/m_pdfS[0];
                                    if(x.mesh->getBSDF()->isDiffuse()) {
                                        w += pcur;
                                    }

                                    /* backward, in direction from light to eye */
                                    pcur = 1;
                                    if(s==1) xi = (y0_p + y0_n);
                                    else xi = m_itsS[s-2].p;
                                    x = m_itsS[s-1];
                                    xo = m_itsT[t-1].p;
                                    y = m_itsT[t-1];

                                    for(int tt=t; tt>1; tt--){

                                        if(!x.mesh->getBSDF()->isDiffuse()){
                                            pcur *=  G(scene, x, y) / x.shFrame.n.dot((xo - x.p).normalized())  / (m_pdfT[tt-1] * m_GT[tt-1]);
                                        }else{
                                            pcur *= PProjectedSolidAngle(scene, xi, x, xo) * G(scene, x,y) / (m_pdfT[tt-1] * m_GT[tt-1]);
                                        }

                                        if(m_itsT[tt-1].mesh->getBSDF()->isDiffuse()
                                           && m_itsT[tt-2].mesh->getBSDF()->isDiffuse()){
                                            w += pcur;
                                        }

                                        /* update */
                                        xi = x.p;
                                        x = m_itsT[tt-1];
                                        xo = m_itsT[tt-2].p;
                                        y = m_itsT[tt-2];
                                    }

                                    w = 1 / w;
                                    if(w!=w) w=0;

                                }
                                rad += m_alphaS[s] * FrS * g * FrT * m_alphaT[t] * w;
                            }
                        }
                    }
                }
            }
            return rad;
        }


        float PProjectedSolidAngle(const Scene *scene, Point3f xi, Intersection x, Point3f xo) const{
            BSDFQueryRecord bRec = BSDFQueryRecord(x.shFrame.toLocal((xi-x.p).normalized()));
            Vector3f wo =(xo-x.p).normalized();
            bRec.wo = x.shFrame.toLocal(wo);
            bRec.measure = ESolidAngle;
            return x.mesh->getBSDF()->pdf(bRec)/x.shFrame.n.dot(wo);
        }

        Color3f Fr(const Scene *scene, Point3f xi, Intersection x, Point3f xo) const{
            BSDFQueryRecord bRec = BSDFQueryRecord(x.shFrame.toLocal((xi-x.p).normalized()));
            bRec.wo = x.shFrame.toLocal((xo-x.p).normalized());
            bRec.measure = ESolidAngle;
            return x.mesh->getBSDF()->eval(bRec);
        }

        float G(const Scene *scene, Intersection x, Intersection y) const{
            Intersection t;
            Vector3f wo = y.p - x.p;
            float dis = wo.norm();
            wo.normalize();

            if(!scene->rayIntersect(Ray3f(x.p, wo), t)
                    || std::abs(t.t - dis)>0.0001)
                return 0.0f;

            float g = x.shFrame.n.dot(wo);            if(g<0) return 0.0f;
            g*=y.shFrame.n.dot(-wo);                  if(g<0) return 0.0f;
            g/=dis*dis;

            return g;
        }

        bool generatePath(const Scene *scene, Sampler *sampler, Intersection x, Vector3f wi,
                          std::vector<Intersection> &its, std::vector<float> &pdf, std::vector<float> &g) const {

            int depth = 5;
            float SRVL = 0.9f, pSRVL, mpdf=1, mg=1;

            Intersection y;
            BSDFQueryRecord bRec = BSDFQueryRecord(x.toLocal(wi));
            bRec.measure = ESolidAngle;
            const BSDF *bsdf = nullptr;

            while (true) {
                its.push_back(x);
                pdf.push_back(mpdf);
                g.push_back(mg);

                /* depth and the Russian roulette */
                if (depth-- > 0) pSRVL = 1;
                else if (sampler->next1D() > SRVL) return true;
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
            }

        }

        std::string toString() const {
            return "BiDirectional Path Tracing Integrator[]";
        }

    };


    NORI_REGISTER_CLASS(BDPTIntegrator, "bdpt");
NORI_NAMESPACE_END
