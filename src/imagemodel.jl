abstract type AbstractImagingModel end

using StatsFuns: logistic
abstract type PolRep end
struct Poincare <: PolRep end
struct ExpMap <: PolRep end
struct TotalIntensity <: PolRep end
struct Matern end

struct ImagingModel{P,M,G,F,B,AG,C} <: AbstractImagingModel
    mimg::M
    grid::G
    ftot::F
    base::B
    order::Int
end

function ImagingModel(p::PolRep, mimg::M, grid, ftot; order=1, base=GMRF, center=centerfix(M)) where {M}
    b = prepare_base(base, grid, order)
    return ImagingModel{typeof(p),M,typeof(grid),typeof(ftot),typeof(b), center}(mimg, grid, ftot, b, order)
end

function ImagingModel(p::PolRep, mimg::IntensityMap, ftot; order=1, base=GMRF, center=centerfix(typeof(mimg)))
    return ImagingModel(p, mimg./sum(mimg), axisdims(mimg), ftot; order=order, base=base, center)
end


prepare_base(b::VLBIImagePriors.MarkovRandomField, grid, order) = b
prepare_base(::Matern, grid, order) = first(matern(size(grid)))



center(::ImagingModel{P, M, G, F, B, AG, C}) where {P, M, G, F, B, AG, C} = C

getftot(m::ImagingModel{P, M, G, <:Real}, _)   where {P, M, G} = m.ftot
getftot(::ImagingModel{P, M, G}, θ) where {P, M, G} = θ.ftot

function (m::ImagingModel{P})(θ, meta) where {P}
    mimg = make_mean(m.mimg, m.grid, θ)
    fimg = getftot(m, θ)

    pmap = make_image(P, m.base, fimg, mimg, θ)
    if center(m)
        x0, y0 = centroid(pmap)
        return shifted(ContinuousImage(pmap, BSplinePulse{3}()), -x0, -y0)
    else
        return ContinuousImage(pmap, BSplinePulse{3}())
    end

end

function make_image(::Type{<:Poincare}, ::VLBIImagePriors.MarkovRandomField, ftot, mimg, θ)
    (;c, σ, p, p0, pσ, angparams) = θ
    return make_poincare(ftot, mimg, σ*c.params, p0, pσ, p.params, angparams)
end

function make_image(::Type{<:ExpMap}, ::VLBIImagePriors.MarkovRandomField, ftot, mimg, θ)
    (;a, b, c, d, σa, σb, σc, σd,) = θ
    aimg = similar(a.params)
    bimg = similar(b.params)
    cimg = similar(c.params)
    dimg = similar(d.params)
    
    @inbounds for i in eachindex(a, b, c, d)
        aimg[i] = σa*a.params[i]
        bimg[i] = σb*b.params[i]
        cimg[i] = σc*c.params[i]
        dimg[i] = σd*d.params[i]
    end
    return make_pol2expimage(ftot, aimg, bimg, cimg, dimg, mimg)
end


function make_image(::Type{<:Poincare}, trf::VLBIImagePriors.StationaryMatern, ftot, mimg, θ)
    (;c, σ, ρ, ν, p, p0, pσ, pν, pρ, angparams) = θ
    δ = trf(c, ρ, ν)
    pδ= trf(p, pρ, pν)
    for i in eachindex(δ)
        δ[i] *= σ
    end
    return make_poincare(ftot, mimg, δ, p0, pσ, pδ, angparams)
end

function make_image(::Type{<:ExpMap}, trf::VLBIImagePriors.StationaryMatern, ftot, mimg, θ)
    (;a, b, c, d, ρa, ρb, ρc, ρd, νa, νb, νc, νd, σa, σb, σc, σd,) = θ
    δa = trf(a, ρa, νa)
    δb = trf(b, ρb, νb)
    δc = trf(c, ρc, νc)
    δd = trf(d, ρd, νd)
    @inbounds for i in eachindex(δa, δb, δc, δd)
        δa[i] *= σa
        δb[i] *= σb
        δc[i] *= σc
        δd[i] *= σd
    end 
    return make_pol2expimage(ftot, δa, δb, δc, δd, mimg)
end

function make_image(::Type{<:TotalIntensity}, ::VLBIImagePriors.MarkovRandomField, ftot, mimg, θ)
    (;c, σ) = θ
    img = apply_fluctuations(CenteredLR(), mimg, σ*c.params)
    bimg = baseimage(img)
    for i in eachindex(img)
        bimg[i] *= ftot
    end
    return  img
end

function make_image(::Type{<:TotalIntensity}, ::VLBIImagePriors.StationaryMatern, ftot, mimg, θ)
    (;c, σ, ρ, ν) = θ
    δ = trf(c, ρ, ν)
    δ .*= σ
    img = apply_fluctuations(CenteredLR(), mimg, δ)
    bimg = baseimage(img)
    for i in eachindex(img)
        bimg[i] *= ftot
    end
    return img
end


function make_poincare(ftot, mimg, δ, p0, pσ, pδ, angparams)
    stokesi = apply_fluctuations(CenteredLR(), mimg, δ)
    pstokesi = parent(stokesi)
    for i in eachindex(pstokesi)
        pstokesi[i] *= ftot
    end
    ptotim  = logistic.(p0 .+ pσ.*pδ)
    pmap = PoincareSphere2Map(stokesi, ptotim, angparams)
    return pmap
end

function make_pol2expimage(ftot, a, b, c, d, mimg)
    δ = PolExp2Map(a, b, c, d, axisdims(mimg))
    bδ = baseimage(δ)
    δI= sum(bδ.I)
    brast = baseimage(mimg).*bδ./δI
    brast .*= ftot/sum(brast.I)
    pmap = IntensityMap(brast, axisdims(mimg)) 
    return pmap
end

centerfix(::Type{<:Any}) = true

function skyprior(m::ImagingModel{P}; beamsize=μas2rad(20.0), overrides::Dict=Dict()) where {P}
    imgprior = genimgprior(P, m.base, m.grid, beamsize, m.order)
    mprior = genmeanprior(m.mimg)

    if !(m.ftot isa Real)
        imgprior[:ftot] = m.ftot
    end

    prior = merge(imgprior, mprior)

    for k in keys(overrides)
        default[k] = overrides[k]
    end


    return NamedTuple(prior)
end

function genimgprior(::Type{<:Poincare}, base::VLBIImagePriors.MarkovRandomField, grid, beamsize, order)
    cprior = corr_image_prior(grid, beamsize; base=base, order=order)
    default = Dict(
        :c => cprior,
        :σ => truncated(Normal(0.0, 0.5); lower = 0.0),
        :p => cprior,
        :p0=> Normal(-1.0, 2.0),
        :pσ=> truncated(Normal(0.0, 0.5); lower = 0.0),
        :angparams => ImageSphericalUniform(size(cprior.priormap.cache)...)
        )
    return default
end

function genimgprior(::Type{<:ExpMap}, base::VLBIImagePriors.MarkovRandomField, grid, beamsize, order)
    cprior = corr_image_prior(grid, beamsize; base=base, order=order)
    default = Dict(
        :a => cprior,
        :b => cprior,
        :c => cprior,
        :d => cprior,
        :σa => truncated(Normal(0.0, 0.5); lower=0.0),
        :σb => truncated(Normal(0.0, 0.5); lower=0.0),
        :σc => truncated(Normal(0.0, 0.5); lower=0.0),
        :σd => truncated(Normal(0.0, 0.05); lower=0.0),
        )
    return default
end

function genimgprior(::Type{<:Poincare}, base::VLBIImagePriors.StationaryMatern, grid, beamsize, order)
    bs = beamsize/pixelsizes(grid).X
    cprior = VLBIImagePriors.std_dist(base)
    default = Dict(
        :c  => cprior,
        :σ  => truncated(Normal(0.0, 0.5); lower = 0.0),
        :ρ  => 0.5+InverseGamma(1.0, -log(0.1)*bs),
        :ν  => 0.1+InverseGamma(5.0, 9.0), 
        :p  => cprior,
        :ρp => 0.5+InverseGamma(1.0, -log(0.1)*bs),
        :νp => 0.1+InverseGamma(5.0, 9.0), 
        :p0 => Normal(-1.0, 2.0),
        :pσ => truncated(Normal(0.0, 0.5); lower = 0.0),
        :angparams => ImageSphericalUniform(size(cprior.priormap.cache)...)
        )
    return default
end

function genimgprior(::Type{<:ExpMap}, base::VLBIImagePriors.StationaryMatern, grid, beamsize, order)
    bs = beamsize/pixelsizes(grid).X
    cprior = VLBIImagePriors.std_dist(base)
    default = Dict(
        :a => cprior,
        :b => cprior,
        :c => cprior,
        :d => cprior,
        :σa => truncated(Normal(0.0, 0.5); lower=0.0),
        :σb => truncated(Normal(0.0, 0.5); lower=0.0),
        :σc => truncated(Normal(0.0, 0.5); lower=0.0),
        :σd => truncated(Normal(0.0, 0.1); lower=0.0),
        :ρa    => 0.5+InverseGamma(1.0, -log(0.1)*bs),
        :νa    => 0.1+InverseGamma(5.0, 9.0), 
        :ρb    => 0.5+InverseGamma(1.0, -log(0.1)*bs),
        :νb    => 0.1+InverseGamma(5.0, 9.0),
        :ρc    => 0.5+InverseGamma(1.0, -log(0.1)*bs),
        :νc    => 0.1+InverseGamma(5.0, 9.0),
        :ρd    => 0.5+InverseGamma(1.0, -log(0.1)*bs),
        :νd    => 0.1+InverseGamma(5.0, 9.0)
        )
    return default
end

function genimgprior(::Type{<:TotalIntensity}, base::VLBIImagePriors.MarkovRandomField, grid, beamsize, order)
    cprior = corr_image_prior(grid, beamsize; base=base, order=order)
    default = Dict(
        :c => cprior,
        :σ => truncated(Normal(0.0, 0.5); lower = 0.0)
        )
    return default
end

function genimgprior(::Type{<:TotalIntensity}, base::VLBIImagePriors.StationaryMatern, grid, beamsize, order)
    bs = beamsize/pixelsizes(grid).X
    cprior = VLBIImagePriors.std_dist(base)
    default = Dict(
        :c => cprior,
        :σ => truncated(Normal(0.0, 0.5); lower = 0.0),
        :ρ => 0.5+InverseGamma(1.0, -log(0.1)*bs),
        :ν => 0.1+InverseGamma(5.0, 9.0)
        )
    return default
end


function make_mean(mimg::IntensityMap, grid, θ)
    return mimg
end

function genmeanprior(::IntensityMap)
    return Dict()
end

struct MimgPlusBkg{M}
    mimg::M
end

function make_mean(mimg::MimgPlusBkg, grid, θ)
    (;fb) = θ
    fbn = fb/(prod(size(grid)))
    return mimg.mimg.*((1-fb)) .+ fbn
end

function genmeanprior(::MimgPlusBkg)
    return Dict(:fb => Beta(1.0, 5.0))
end


struct DblRing end
centerfix(::Type{<:DblRing}) = false

function make_mean(::DblRing, grid, θ)
    (;r0, ain, aout,) = θ
    m = modify(RingTemplate(RadialDblPower(ain, aout), AzimuthalUniform()), Stretch(r0))
    mimg = intensitymap(m, gmrf.g)
    pmimg = baseimage(mimg)
    pmimg .= pmimg./sum(pmimg)
    return mimg 
end

function genmeanprior(::DblRing)
    return Dict(
        :r0        => Uniform(μas2rad(15.0), μas2rad(25.0)),
        :ain       => Exponential(1.0)+3,
        :aout      => Exponential(1.0)+1
        )
end