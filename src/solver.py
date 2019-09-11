# -*- coding: utf-8 -*-
"""
    solver
    ~~~~~~

    Logistic regression solver.
"""
import numpy as np
import utils


class BinaryLogisticRegression:
    """Logistic regression with trimming"""
    def __init__(self,
                 idata,
                 lam=0.1,
                 inlier_pct=1.0):
        # pass in the data
        self.idata = idata
        self.lam = lam
        self.inlier_pct = inlier_pct

        # create optimization variables
        self.m = self.idata.num_images
        self.k = np.prod(self.idata.image_shape)
        self.A = self.idata.images
        self.y = self.idata.labels
        self.y[self.idata.class_slices[0]] = -1.0
        self.y[self.idata.class_slices[1]] = 1.0

        self.h = np.floor(self.inlier_pct*self.m)
        self.inlier_pct = self.h/self.m
        self.w = np.repeat(self.inlier_pct, self.m)

        self.use_trimming = self.inlier_pct != 1.0

    def objective(self, x):
        """objective function"""
        r = -self.y*self.A.dot(x)
        val = self.w.dot(np.log(1.0 + np.exp(r)))/self.h
        val += 0.5*self.lam*np.sum(x**2)
        return val

    def gradient(self, x):
        """gradient function"""
        r = -self.y*self.A.dot(x)
        grad = -self.A.T.dot(self.w*self.y*np.exp(r)/(1.0 + np.exp(r)))/self.h
        grad += self.lam*x
        return grad

    def hessian(self, x):
        """hessian function"""
        r = self.y*self.A.dot(x)
        hess = (self.A.T*(self.w*np.exp(r)/
                          (1.0 + np.exp(r))**2)).dot(self.A)/self.h
        hess += self.lam*np.eye(self.k)
        return hess

    def gradient_trimming_weights(self, x, normalize_grad=True):
        """gradient w.r.t. the trimming weights"""
        r = -self.y*self.A.dot(x)
        grad = np.log(1.0 + np.exp(r))/self.h
        if normalize_grad:
            grad /= np.max(np.abs(grad))
        return grad

    def update_trimming_weights(self, x, step_size, normalize_grad=True):
        """update trimming weight with given step size"""
        w_new = utils.project_onto_capped_simplex(
                self.w - step_size*self.gradient_trimming_weights(
                        x, normalize_grad=normalize_grad),
                self.h)
        return w_new

    def fit_model(self,
                  x0=None,
                  max_iter=100,
                  tol=1e-6,
                  verbose=False,
                  trimming_step_size=500.0,
                  trimming_normalize_grad=True):
        if x0 is None:
            x0 = np.zeros(self.k)

        x = x0.copy()
        obj = self.objective(x)
        err = tol + 1.0
        iter_count = 0

        if verbose:
            print("initial obj: %7.2e" % obj)

        while err >= tol:
            # newton's step on x
            x_new = x - np.linalg.solve(self.hessian(x), self.gradient(x))
            # trimming step
            if self.use_trimming:
                w_new = self.update_trimming_weights(
                        x_new,
                        trimming_step_size,
                        normalize_grad=trimming_normalize_grad)

            # update information
            obj = self.objective(x_new)
            err = np.linalg.norm(self.gradient(x_new))
            if self.use_trimming:
                err += np.linalg.norm(w_new - self.w)
            np.copyto(x, x_new)
            if self.use_trimming:
                np.copyto(self.w, w_new)

            iter_count += 1
            if verbose:
                print("iter %i, obj %7.2e, err %7.2e" %
                      (iter_count, obj, err))

            # check stop criterion
            if iter_count >= max_iter:
                if verbose:
                    print("reach maximum number of iterations.")
                break

        classifier = utils.BinaryImageClassifier(x, self.idata.image_shape)
        if self.use_trimming:
            outlier_id = self.w < 0.5
            outliers = utils.ImageData(
                    self.idata.images[outlier_id],
                    self.idata.image_shape,
                    labels=self.idata.labels[outlier_id])
        else:
            outliers = None

        return classifier, outliers
