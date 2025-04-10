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
  id 1840
  arrival_time 40625.970605167626
  lifetime 1859.1820851322502
  num_nodes 5
  type "path"
  node [
    id 0
    label "0"
    cpu 41
    gpu 44
    rom 3
  ]
  node [
    id 1
    label "1"
    cpu 33
    gpu 39
    rom 25
  ]
  node [
    id 2
    label "2"
    cpu 41
    gpu 38
    rom 34
  ]
  node [
    id 3
    label "3"
    cpu 12
    gpu 10
    rom 40
  ]
  node [
    id 4
    label "4"
    cpu 25
    gpu 22
    rom 1
  ]
  edge [
    source 0
    target 1
    bw 28
  ]
  edge [
    source 1
    target 2
    bw 19
  ]
  edge [
    source 2
    target 3
    bw 41
  ]
  edge [
    source 3
    target 4
    bw 28
  ]
]
