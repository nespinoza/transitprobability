import numpy as np

def get_quantiles(dist,alpha = 0.68, method = 'median'):
    """ 
    get_quantiles function

    DESCRIPTION

        This function returns, in the default case, the parameter median and the error% 
        credibility around it. This assumes you give a non-ordered 
        distribution of parameters.

    OUTPUTS

        Median of the parameter,upper credibility bound, lower credibility bound

    """
    ordered_dist = dist[np.argsort(dist)]
    param = 0.0 
    # Define the number of samples from posterior
    nsamples = len(dist)
    nsamples_at_each_side = int(nsamples*(alpha/2.)+1)
    if(method == 'median'):
       med_idx = 0 
       if(nsamples%2 == 0.0): # Number of points is even
          med_idx_up = int(nsamples/2.)+1
          med_idx_down = med_idx_up-1
          param = (ordered_dist[med_idx_up]+ordered_dist[med_idx_down])/2.
          return param,ordered_dist[med_idx_up+nsamples_at_each_side],\
                 ordered_dist[med_idx_down-nsamples_at_each_side]
       else:
          med_idx = int(nsamples/2.)
          param = ordered_dist[med_idx]
          return param,ordered_dist[med_idx+nsamples_at_each_side],\
                 ordered_dist[med_idx-nsamples_at_each_side]

def sample_from_errors(loc, sigma1, sigma2, n, low_lim = None, up_lim = None):
    """
    Description
    -----------
    Function made to sample points given the 0.16, 0.5 and 0.84 quantiles of a parameter
    In the case of unequal variances, this algorithm assumes a skew-normal distribution and samples from it. 
    If the variances are equal, it samples from a normal distribution.
    Inputs
    ------
          loc:           Location parameter (we hope it is the 0.5 quantile, i.e., the median).
       sigma1:           Upper error of the parameter (we hope loc+sigma1 is the 0.84 quantile, i.e., the "upper 1-sigma bound").
 
       sigma2:           Lower error of the parameter (we hope loc-sigma2 is the 0.16 quantile, i.e., the "lower 1-sigma bound").
            n:           Number of samples you want to generate from this.
      low_lim:           (Optional) Lower limits on the values of the samples*.
       up_lim:           (Optional) Upper limits on the values of the samples*.
    Outputs
    -------
        The output are n samples from the distribution that best-matches the quantiles.
    *The optional inputs (low_lim and up_lim) are lower and upper limits that the samples have to have; if any of the samples 
    surpasses those limits, new samples are drawn until no samples do. Note that this changes the actual variances of the samples.
    """

    if (sigma1 != sigma2):
            """
             If errors are assymetric, sample from a skew-normal distribution given
             the location parameter (assumed to be the median), sigma1 and sigma2.
            """

            # First, find the parameters mu, sigma and alpha of the skew-normal distribution that 
            # best matches the observed quantiles:
            sknorm = skew_normal()
            sknorm.fit(loc,sigma1,sigma2)

            # And now sample n values from the distribution:
            samples = sknorm.sample(n)

            # If a lower limit or an upper limit is given, then search if any of the samples surpass 
            # those limits, and sample again until no sample surpasses those limits:
            if low_lim is not None:
               while True:
                     idx = np.where(samples<low_lim)[0]
                     l_idx = len(idx)
                     if l_idx > 0:
                        samples[idx] = sknorm.sample(l_idx)
                     else:
                        break
            if up_lim is not None:
               while True:
                     idx = np.where(samples>up_lim)[0]
                     l_idx = len(idx)
                     if l_idx > 0:
                        samples[idx] = sknorm.sample(l_idx)
                     else:
                        break
            return samples       

    else:
            """
             If errors are symmetric, sample from a gaussian
            """
            samples = np.random.normal(loc,sigma1,n)
            # If a lower limit or an upper limit is given, then search if any of the samples surpass 
            # those limits, and sample again until no sample surpasses those limits:
            if low_lim is not None:
               while True:
                     idx = np.where(samples<low_lim)[0]
                     l_idx = len(idx)
                     if l_idx > 0:
                        samples[idx] = np.random.normal(loc,sigma1,l_idx)
                     else:
                        break
            if up_lim is not None:
               while True:
                     idx = np.where(samples>up_lim)[0]
                     l_idx = len(idx)
                     if l_idx > 0:
                        samples[idx] = np.random.normal(loc,sigma1,l_idx)
                     else:
                        break
            return samples
           
from scipy.integrate import quad
from scipy.optimize import leastsq

class skew_normal: 
      """
      Description
      -----------
      This class defines a skew_normal object, which generates a skew_normal distribution given the quantiles 
      from which you can then sample datapoints from.
      """
      def __init__(self):
         self.mu = 0.0
         self.sigma = 0.0
         self.alpha = 0.0

      def fit(self, median, sigma1, sigma2):
            """
            This function fits a Skew Normal distribution given
            the median, upper error bars (sigma1) and lower error bar (sigma2).
            """

            # First, define the sign of alpha, which should be positive if right skewed
            # and negative if left skewed:
            alpha_sign = np.sign(sigma1-sigma2)

            # Now define the residuals of the least-squares problem:
            def residuals(p, data, x):
                mu, sqrt_sigma, sqrt_alpha = p
                return data - model(x, mu, sqrt_sigma, sqrt_alpha)

            # Define the model used in the residuals:
            def model(x, mu, sqrt_sigma, sqrt_alpha):
                """
                Note that we pass the square-root of the scale (sigma) and shape (alpha) parameters, 
                in order to define the sign of the former to be positive and of the latter to be fixed given
                the values of sigma1 and sigma2:
                """
                return self.cdf (x, mu, sqrt_sigma**2, alpha_sign * sqrt_alpha**2)

            # Define the quantiles:
            y = np.array([0.15866, 0.5, 0.84134])

            # Define the values at which we observe the quantiles:
            x = np.array([median - sigma2, median, median + sigma1])

            # Start assuming that mu = median, sigma = mean of the observed sigmas, and alpha = 0 (i.e., start from a gaussian):
            guess = (median, np.sqrt ( 0.5 * (sigma1 + sigma2) ), 0)
            # Perform the non-linear least-squares optimization:
            plsq = leastsq(residuals, guess, args=(y, x))[0]

            self.mu, self.sigma, self.alpha = plsq[0], plsq[1]**2, alpha_sign*plsq[2]**2

      def sample(self, n):
            """
            This function samples n points from a skew normal distribution using the 
            method outlined by Azzalini here: http://azzalini.stat.unipd.it/SN/faq-r.html.
            """
            # Define delta:
            delta = self.alpha/np.sqrt(1+self.alpha**2)
            
            # Now sample u0,u1 having marginal distribution ~N(0,1) with correlation delta:
            u0 = np.random.normal(0,1,n)
            v = np.random.normal(0,1,n)
            u1 = delta*u0 + np.sqrt(1-delta**2)*v

            # Now, u1 will be random numbers sampled from skew-normal if the corresponding values 
            # for which u0 are shifted in sign. To do this, we check the values for which u0 is negative:
            idx_negative = np.where(u0<0)[0]
            u1[idx_negative] = -u1[idx_negative]

            # Finally, we change the location and scale of the generated random-numbers and return the samples:
            return self.mu + self.sigma*u1

      @staticmethod
      def cdf(x, mu, sigma, alpha):
            """
            This function simply calculates the CDF at x given the parameters 
            mu, sigma and alpha of a Skew-Normal distribution. It takes values or 
            arrays as inputs.
            """
            if type(x) is np.ndarray:
               out = np.zeros(len(x))
               for i in range(len(x)):
                   out[i] = quad(lambda x: skew_normal.pdf(x,mu,sigma,alpha), -np.inf, x[i])[0]
               return out

            else:
               return quad(lambda x: skew_normal.pdf(x,mu,sigma,alpha), -np.inf, x)[0]

      @staticmethod
      def pdf(x, mu, sigma, alpha):
            """
            This function returns the value of the Skew Normal PDF at x, given
            mu, sigma and alpha
            """
            def erf(x):
                # save the sign of x
                sign = np.sign(x)
                x = abs(x)

                # constants
                a1 =  0.254829592
                a2 = -0.284496736
                a3 =  1.421413741
                a4 = -1.453152027
                a5 =  1.061405429
                p  =  0.3275911

                # A&S formula 7.1.26
                t = 1.0/(1.0 + p*x)
                y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
                return sign*y

            def palpha(y,alpha):
                phi = np.exp(-y**2./2.0)/np.sqrt(2.0*np.pi)
                PHI = ( erf(y*alpha/np.sqrt(2)) + 1.0 )*0.5
                return 2*phi*PHI

            return palpha((x-mu)/sigma,alpha)*(1./sigma)
