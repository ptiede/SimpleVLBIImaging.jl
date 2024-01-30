
"""
    MeanModels

This specifies the mean model that will be used for an imaging problem.
The user must subtype this model and then make a functor that returns the
mean image in the log-ratio space using `to_real(CenteredLR(), mean/flux(mean))`
"""
abstract type MeanModels end

"""
    AdaptiveMean(f, grid::RectiGrid)

Uses an adaptive mean that is defined by the functor `f` on the grid `grid`. The function
`f` must take in a named tuple of paramters and return a `VLBISkyModels.AbstractModel` subtype.

## Example

```julia
f(p) = modify(Gaussian(), Stretch(p.σ, p.σ*(1+p.τ)), Rotate(p.ξ))
g = imagepixels(μas2rad(150), μas2rad(150), 256, 256)
m = AdaptiveMean(f, g)
mimg = m((σ = μas2rad(20.0), τ = 0.5, ξ = π/3))
```
"""
struct AdaptiveMean{F, G} <: MeanModels
    f::F
    grid::G
end

function (model::AdaptiveMean)(x)
    img = intensitymap(model.f(x), model.grid) |> baseimage
    mr = to_real(CenteredLR(), img/sum(img))
    return mr
end

function (model::SimpleMean)(x)
    return to_real(CenteredLR(), (model.gauss .+ x.fb*model.bkgd)./(1+x.fb))
end

struct SimpleMean{G, B} <: MeanModels
    gauss::G
    bkgd::B
    """
        SimpleMean(fwhm::Real, grid::RectiGrid)

    Uses a simple mean that is a circular Gaussian with FWHM `fwhm` defined on the
    grid `grid`. A second Gaussian is added with FWHM `fovx/2` and `fovy/2` and
    and whose relative value is dynamically determined by the parameter `fb`, in
    the functor call.

    ## Example

    ```julia
    g = imagepixels(μas2rad(150), μas2rad(150), 256, 256)
    m = SimpleMean(μas2rad(50), g)
    mimg = m((fb = 0.5,))
    ```
    """
    function SimpleMean(fwhm::Real, grid::RectiGrid)
        fovx, fovy = fieldofview(grid)
        fwhmfac = 2*sqrt(2*log(2))
        m = modify(Gaussian(), Stretch(fwhm/fwhmfac))
        bkgd_m = modify(Gaussian(), Stretch(fovx/2, fovy/2), Renormalize(1.0))
        gauss = intensitymap(m, grid)
        gauss ./= flux(gauss)
        bkgd = intensitymap(bkgd_m, grid)
        bkgd ./= flux(bkgd)
        gauss_base = baseimage(gauss)
        bkgd_base = baseimage(bkgd)
        return new{typeof(gauss_base), typeof(bkgd_base)}(gauss_base, bkgd_base)
    end
end

struct FixedMean{G} <: MeanModels
    rimg::G
    """
        FixedMean(m::VLBISkyModels.AbstractModel, grid::RectiGrid)

    Use a mean image that is fixed and is given by the image `m` on the grid `grid`.
    """
    function FixedMean(m::VLBISkyModels.AbstractModel, grid::RectiGrid)
        img = intensitymap(m, grid)
        rimg = to_real(CenteredLR(), img/flux(img)) |> baseimage
        return new{typeof(rimg)}(rimg)
    end
end

@inline (model::FixedMean)(x) = model.rimg
