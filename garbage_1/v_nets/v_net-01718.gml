graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 1718
  arrival_time 38291.420657739036
  lifetime 48.184361306020385
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 14
    gpu 0
    rom 9
  ]
  node [
    id 1
    label "1"
    cpu 47
    gpu 10
    rom 38
  ]
  node [
    id 2
    label "2"
    cpu 33
    gpu 9
    rom 25
  ]
  node [
    id 3
    label "3"
    cpu 50
    gpu 6
    rom 10
  ]
  node [
    id 4
    label "4"
    cpu 31
    gpu 16
    rom 40
  ]
  node [
    id 5
    label "5"
    cpu 42
    gpu 33
    rom 32
  ]
  node [
    id 6
    label "6"
    cpu 10
    gpu 42
    rom 31
  ]
  node [
    id 7
    label "7"
    cpu 13
    gpu 27
    rom 40
  ]
  node [
    id 8
    label "8"
    cpu 25
    gpu 24
    rom 35
  ]
  node [
    id 9
    label "9"
    cpu 37
    gpu 3
    rom 26
  ]
  node [
    id 10
    label "10"
    cpu 30
    gpu 9
    rom 22
  ]
  edge [
    source 0
    target 1
    bw 44
  ]
  edge [
    source 1
    target 2
    bw 2
  ]
  edge [
    source 2
    target 3
    bw 43
  ]
  edge [
    source 3
    target 4
    bw 38
  ]
  edge [
    source 4
    target 5
    bw 11
  ]
  edge [
    source 5
    target 6
    bw 0
  ]
  edge [
    source 6
    target 7
    bw 40
  ]
  edge [
    source 7
    target 8
    bw 19
  ]
  edge [
    source 8
    target 9
    bw 21
  ]
  edge [
    source 9
    target 10
    bw 14
  ]
]
