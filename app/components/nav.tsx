import Link from 'next/link'

const navItems = {
  '/': {
    name: 'home',
  },
  '/apps': {
    name: 'apps',
  },
  '/blog': {
    name: 'blog',
  },
  '/papers': {
    name: 'papers',
  },
  '/projects': {
    name: 'projects',
  },
}

export function Navbar() {
  return (
    <aside className="-ml-[8px] mb-16 tracking-tight">
      <div className="lg:sticky lg:top-20">
        <nav
          className="flex flex-row items-start relative px-0 pb-0 fade md:overflow-auto scroll-pr-6 md:relative"
          id="nav"
        >
          <div className="flex flex-row space-x-0 pr-10">
            <Link
              key="/"
              href="/"
              className="transition-all hover:text-neutral-800 dark:hover:text-neutral-200 flex align-middle relative py-1 px-2 m-1"
            >
              home
            </Link>
            {Object.entries(navItems)
              .filter(([path]) => path !== '/')
              .sort((a, b) => a[1].name.localeCompare(b[1].name))
              .map(([path, { name }]) => {
                return (
                  <Link
                    key={path}
                    href={path}
                    className="transition-all hover:text-neutral-800 dark:hover:text-neutral-200 flex align-middle relative py-1 px-2 m-1"
                  >
                    {name}
                  </Link>
                )
              })}
          </div>
        </nav>
      </div>
    </aside>
  )
}
