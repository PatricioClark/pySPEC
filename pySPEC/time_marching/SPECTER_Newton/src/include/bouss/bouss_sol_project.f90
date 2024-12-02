!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!Project velocity fields!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!Save balance
! INCLUDE 'include/bouss/bouss_global.f90'


!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!Solenoidal projection

! Apply boundary conditions and project onto solenoidal space
   !Option 1: Same as in rkstep2. Added condition in subrout such that o=-1 prevents parameter 
   ! o from modifying values 
   ! o = -1
   ! CALL v_imposebc_and_project(vplanbc,planfc,vx,vy,vz,pr,rki=o,&
   !          v_zsta=(/ vxzsta, vyzsta/), v_zend=(/ vxzend, vyzend/))

   !Option 2: Only use the subroutines from sol_projection, without imposing bc. because this uses
   !pressure field which we do not have for the modified field
   CALL sol_project(vx,vy,vz,pr,1,0,0)

CALL s_imposebc(splanbc,planfc,th)
CALL fc_filter(th)

! ! Hack to prevent weird accumulation in complex domain for th
CALL fftp3d_complex_to_real(planfc,th,R1,MPI_COMM_WORLD)
R1 = R1/nx/ny/nz
CALL fftp3d_real_to_complex(planfc,R1,th,MPI_COMM_WORLD)

!!!!!!!
!Option 3: Go backwards in time and then forwards
! dt = -dt
! DO t = 1,10

!    !Save balance
!    INCLUDE 'include/bouss/bouss_global.f90'

!    !$omp parallel do if (iend-ista.ge.nth) private (j,k)
!    DO i = ista,iend
!    !$omp parallel do if (iend-ista.lt.nth) private (k)
!       DO j = 1,ny
!          DO k = 1,nz

!             INCLUDE 'include/bouss/bouss_rkstep1.f90'

!          END DO
!       END DO
!    END DO

!    ! Runge-Kutta step 2
!    ! Evolves the system in time
!    IF (t.eq.1) THEN
!       DO o = ord,1,-1
!       ! Step 2 of Runge-Kutta for the HD equations
!       ! Computes the nonlinear terms and evolves the equations in dt/o
!                rmp = 1/real(o, kind=GP)

!                ! Non-linear terms
!                CALL gradre(vx,vy,vz,C4,C5,C6)
!                CALL advect(vx,vy,vz,th,C8)

!       !$omp parallel do if (iend-ista.ge.nth) private (j,k)
!                DO i = ista,iend
!       !$omp parallel do if (iend-ista.lt.nth) private (k)
!                   DO j = 1,ny
!                      DO k = 1,nz
!                         C6(k,j,i) = C6(k,j,i) - xmom*th(k,j,i)  ! Bouyancy
!                         C8(k,j,i) = C8(k,j,i) - xtemp*vz(k,j,i) ! Heat current
!                      ENDDO
!                   ENDDO
!                ENDDO

!                ! Dealias non-linear 
!                CALL fc_filter(C4)
!                CALL fc_filter(C5)
!                CALL fc_filter(C6)
!                CALL fc_filter(C8)

!                ! Laplacian
!                CALL laplak(vx,vx)
!                CALL laplak(vy,vy)
!                CALL laplak(vz,vz)
!                CALL laplak(th,th)

!       !$omp parallel do if (iend-ista.ge.nth) private (j,k)
!                DO i = ista,iend
!       !$omp parallel do if (iend-ista.lt.nth) private (k)
!                DO j = 1,ny
!                DO k = 1,nz
!                   ! Perform half oth-RK step
!                   vx(k,j,i) = C1(k,j,i) + dt*(nu*vx(k,j,i)-C4(k,j,i)+&
!                                           fx(k,j,i))*rmp
!                   vy(k,j,i) = C2(k,j,i) + dt*(nu*vy(k,j,i)-C5(k,j,i)+&
!                                           fy(k,j,i))*rmp
!                   vz(k,j,i) = C3(k,j,i) + dt*(nu*vz(k,j,i)-C6(k,j,i)+&
!                                           fz(k,j,i))*rmp
!                   th(k,j,i) = C7(k,j,i) + dt*(kappa*th(k,j,i)-C8(k,j,i)+&
!                                           fs(k,j,i))*rmp
!                ENDDO
!                ENDDO
!                ENDDO

!          CALL sol_project(vx,vy,vz,pr,1,0,0)
!          ! Apply boundary conditions and project onto solenoidal space
!          ! CALL v_imposebc_and_project(vplanbc,planfc,vx,vy,vz,pr,rki=o,&
!                !   v_zsta=(/ vxzsta, vyzsta/), v_zend=(/ vxzend, vyzend/))   

!          CALL s_imposebc(splanbc,planfc,th)
!          CALL fc_filter(th)

!          ! Hack to prevent weird accumulation in complex domain for th
!          CALL fftp3d_complex_to_real(planfc,th,R1,MPI_COMM_WORLD)
!          R1 = R1/nx/ny/nz
!          CALL fftp3d_real_to_complex(planfc,R1,th,MPI_COMM_WORLD)

!       END DO

!    ELSE
!       DO o = ord,1,-1
!          INCLUDE 'include/bouss/bouss_rkstep2.f90'
!       END DO
!    ENDIF

! END DO
   


!Go forwards until present
! dt = -dt
! DO t = 1,10
!    !Save balance
!    INCLUDE 'include/bouss/bouss_global.f90'

!    !$omp parallel do if (iend-ista.ge.nth) private (j,k)
!    DO i = ista,iend
!    !$omp parallel do if (iend-ista.lt.nth) private (k)
!       DO j = 1,ny
!          DO k = 1,nz

!             INCLUDE 'include/bouss/bouss_rkstep1.f90'

!          END DO
!       END DO
!    END DO

!    ! Runge-Kutta step 2
!    ! Evolves the system in time

!    DO o = ord,1,-1
!       INCLUDE 'include/bouss/bouss_rkstep2.f90'
!    END DO

! END DO


! ! !Save balance
! INCLUDE 'include/bouss/bouss_global.f90'

! !!!!!! End Solenoidal Projection
! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!save files: in odir and if saving whole orbit odir newt
WRITE(ext, fmtext) 1 !always save with nmb 1 in odir

!save velocity
rmp = 1.0_GP/ &
   (real(nx,kind=GP)*real(ny,kind=GP)*real(nz,kind=GP))
!$omp parallel do if (iend-ista.ge.nth) private (j,k)
DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
   DO j = 1,ny
      DO k = 1,nz
         C1(k,j,i) = vx(k,j,i)*rmp
         C2(k,j,i) = vy(k,j,i)*rmp
         C3(k,j,i) = vz(k,j,i)*rmp
      END DO
   END DO
END DO
CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
CALL fftp3d_complex_to_real(planfc,C2,R2,MPI_COMM_WORLD)
CALL fftp3d_complex_to_real(planfc,C3,R3,MPI_COMM_WORLD) 
CALL io_write(1,odir,'vx',ext,planio,R1)
CALL io_write(1,odir,'vy',ext,planio,R2)
CALL io_write(1,odir,'vz',ext,planio,R3)

!save pressure
! Pressure. Transform from p'' to p to save
rmp = 1.0_GP/ &
   (real(nx,kind=GP)*real(ny,kind=GP)*dt)
!$omp parallel do if (iend-ista.ge.nth) private (j,k)
DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
   DO j = 1,ny
      DO k = 1,nz
         C1(k,j,i) = pr(k,j,i)*rmp
      END DO
   END DO
END DO
CALL fftp2d_complex_to_real_xy(planfc,C1,R1,MPI_COMM_WORLD)
CALL io_write(1,odir,'pr',ext,planio,R1)

!save th
rmp = 1.0_GP/ &
   (real(nx,kind=GP)*real(ny,kind=GP)*real(nz,kind=GP))
!$omp parallel do if (iend-ista.ge.nth) private (j,k)
DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
   DO j = 1,ny
      DO k = 1,nz
         C1(k,j,i) = th(k,j,i)*rmp
      END DO
   END DO
END DO
CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
CALL io_write(1,odir,'th',ext,planio,R1)