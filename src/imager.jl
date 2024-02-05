function vlbi_imager(prob::ImagingProblem,
               instrument::InstrumentProblem,
               output_folder::String;
               optimization_iters::Int = 1000,
               optimization_trials::Int = 5,
               rng = Random.default_rng(),
               sample::Bool = true,
               nsamples::Int = 5000,
               n_adapts::Int = 2000,
               benchmark::Bool  = true,
               plot_res::Bool = true
               )

    post = build_posterior(prob, instrument)

    tpost = asflat(post)
    ndim = dimension(post)

    if benchmark
        @info "Forward Pass benchmark"
        test = randn(ndim)
        btt = @benchmark logdensityof($tpost, $test)
        io = IOContext(stdout)
        show(io, MIME("text/plain"), btt)
        println()

        @info "Reverse Pass benchmark"
        lt = logdensityof(tpost)
        btt = @benchmark Zygote.gradient($lt, $test)
        io = IOContext(stdout)
        show(io, MIME("text/plain"), btt)
        println()
    end

    sols, ℓopt = best_image(tpost, optimization_trials, optimization_iters, rng)

    out = joinpath(output_folder, "map_image_")

    fov = max(fovx, fovy)
    npix = max(nx, ny)
    for i in 1:(min(5, length(ℓopt)))
        @info "Optimimum $i: $(ℓopt[i])"
        x = Comrade.transform(tpost, sols[i])
        @info "Chi2 2: $(chi2(vlbimodel(post, x), dvis)/(4*2*length(dvis)))"
        img = intensitymap(skymodel(post, x), fov, fov, npix*4, npix*4, x0, y0)
        Comrade.save(out*"$i.fits", img)
        if plot_res
            fig = imageviz(img)
            save(out*"$i.png", fig)
        end
    end

    xopt = Comrade.transform(tpost, sols[begin])

    img = intensitymap(skymodel(post, xopt), fovx, fovy, 2*nx, 2*ny, x0, y0)
    residual(vlbimodel(post, xopt), dvis)
    savefig(out*"residuals.png")

    (;phasecache1, phasecache2, ampcache1, ampcache2, trackcache) = instrumentmeta

    nc = ceil(Int, sqrt(length(stations(dvis))))

    gtpR = Comrade.caltable(phasecache1, xopt.gpR)
    gtpL = Comrade.caltable(phasecache2, xopt.gpr)
    gtR  = Comrade.caltable(ampcache1, exp.(xopt.lgR))
    gtL  = Comrade.caltable(ampcache2, exp.(xopt.lgr))
    dtR  = Comrade.caltable(trackcache, complex.(xopt.dRx, xopt.dRy))
    dtL  = Comrade.caltable(trackcache, complex.(xopt.dLx, xopt.dLy))

    CSV.write(out*"gains_phase_right.csv", gtpR)
    CSV.write(out*"gains_amp_right.csv", gtR)
    CSV.write(out*"gains_phase_ratio.csv", gtpL)
    CSV.write(out*"gains_amp_ratio.csv", gtL)
    CSV.write(out*"dterms_right.csv", dtR)
    CSV.write(out*"dterms_left.csv", dtL)


    if plot_res
        Plots.plot(gtpR, layout=(nc, nc), size=(1200,1200), label="R")
        Plots.plot!(gtpL, layout=(nc,nc), size=(1200,1200), label="Ratio")
        savefig(out*"ctable_phase.png")

        Plots.plot(gtR, layout=(nc,nc), label="R", size=(1200,1200))
        Plots.plot!(gtL, layout=(nc,nc), label="ratio", size=(1200,1200))
        savefig(out*"ctable_amp.png")
    end



    if sample
        out_sample_path = joinpath(output_folder, "SamplingFolder")

        metric = DiagEuclideanMetric(ndim)
        trace = sample(rng, post, AHMC(;metric, autodiff=Val(:Zygote), term_buffer=500),
                       nsamples;
                       saveto=ComradeAHMC.DiskStore(mkpath(out_sample_path), 25),
                       nadapts=nadapt, init_params=xopt
                       )


        chain = load_table(trace, n_adapts+1:100:nsamples)
        c2 = 0.0    ## Construct the image model we fix the flux to 0.6 Jy in this case

        out = joinpath(output_folder, "posterior_")
        if plot_res
            p = residual(vlbimodel(post, chain[end]), dvis)
            for s in sample(chain, 10)
                c2 += chi2(vlbimodel(post, s), dvis)/(4*2*length(dvis))
                residual!(p, vlbimodel(post, s), dvis)
            end
            savefig(out*"residuals.png")
        end

        # Finally let's construct some representative image reconstructions.
        samples = skymodel.(Ref(post), sample(chain, 100))
        fov = max(fovx, fovy)
        npix = max(nx, ny)
        imgs = intensitymap.(samples, fov, fov, npix*2,  npix*2, x0, y0)

        mimg = mean(imgs)
        Comrade.save(out*"mean_image.fits", mimg)
        Comrade.save(out*"random_image.fits", imgs[1])

        if plot_res
            fig = imageviz(mimg)
            save(out*"mean_img.png", fig)
            fig = imageviz(imgs[1])
            save(out*"random_image.png", fig)
        end
    end

    return trace
end
